"""S4 one-wave stable padded slot table stress and isolated old/new timings."""

import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from layer_fixture import require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped, sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.route import (
    _local_slots,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.slot_table import table
from sglang.srt.rocm_muse_spark.muse_moe_triton import moe_align_block_size_small


@g.jit
def run(Ids, Sorted, Experts, NumPost, Ticks, START: gl.constexpr, OLD: gl.constexpr):
    i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]))
    scoped.store_words(Sorted + i, gl.full_like(i, 64), i >= 0)
    scoped.store_words(Experts + gl.minimum(i, 63), gl.full_like(i, -1), i < 64)
    gl.barrier()
    t = sync.clock()
    if OLD:
        _local_slots(Ids, Sorted, Experts, NumPost, START)
    else:
        table(Ids, Sorted, Experts, NumPost, START)
    scoped.drain_stores()
    end = sync.clock()
    gl.store(Ticks, end - t)


def main():
    require_gpu0()
    gen = torch.Generator().manual_seed(20261117)
    patterns = [
        torch.randperm(256, generator=gen)[:64].reshape(8, 8).int(),
        torch.arange(64).reshape(8, 8).int(),
        torch.full((8, 8), 256, dtype=torch.int32),
    ]
    for hits in (1, 8, 16, 17, 32, 33, 64):
        ids = torch.full((64,), 256, dtype=torch.int32)
        ids[:hits] = 3
        patterns.append(ids[torch.randperm(64, generator=gen)].reshape(8, 8))
    for seed in range(20):
        patterns.append(torch.randint(0, 257, (8, 8), generator=gen, dtype=torch.int32))
    sorted_ids = torch.empty(1024, device="cuda", dtype=torch.int32)
    experts = torch.empty(64, device="cuda", dtype=torch.int32)
    count = torch.empty(1, device="cuda", dtype=torch.int32)
    ticks = torch.empty(1, device="cuda", dtype=torch.int64)
    rows = []
    for case, host_ids in enumerate(patterns):
        ids = host_ids.cuda()
        for rank in range(4):
            call = lambda: run[(1,)](
                ids,
                sorted_ids,
                experts,
                count,
                ticks,
                rank * 64,
                False,
                num_warps=4,
                enable_fp_fusion=False,
            )
            kernel = call()
            torch.cuda.synchronize()
            expected = moe_align_block_size_small(ids, 64, rank * 64, 16)
            assert all(
                torch.equal(a, b)
                for a, b in zip((sorted_ids, experts, count), expected)
            ), (case, rank)
            if case == 0 and rank == 0:
                old = lambda: run[(1,)](
                    ids,
                    sorted_ids,
                    experts,
                    count,
                    ticks,
                    0,
                    True,
                    num_warps=16,
                    enable_fp_fusion=False,
                )
                timing = graph_times(call)
                call()
                torch.cuda.synchronize()
                newticks = int(ticks.item())
                oldtime = graph_times(old)
                old()
                torch.cuda.synchronize()
                oldticks = int(ticks.item())
                row = dict(
                    cases=len(patterns) * 4,
                    new=timing,
                    old=oldtime,
                    new_ticks=newticks,
                    old_ticks=oldticks,
                    vgprs=kernel.n_regs,
                    lds=kernel.metadata.shared,
                )
                rows.append(row)
                for fmt in ("amdgcn", "ttgir"):
                    Path(f"/home/bangtliu/coord/logs/phase4-S4-slots.{fmt}").write_text(
                        kernel.asm[fmt]
                    )
    Path("/home/bangtliu/coord/logs/phase4-S4-slots.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(
        json.dumps(dict(event="PASS", comparisons=len(patterns) * 4, timing=rows)),
        flush=True,
    )


if __name__ == "__main__":
    main()

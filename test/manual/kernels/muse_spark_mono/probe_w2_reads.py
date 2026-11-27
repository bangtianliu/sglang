"""Bounded persistent W2 A-read comparison after a scoped in-kernel publication."""

import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from layer_fixture import require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped, sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_bf16_tile,
    store_tile_words,
)

ROOT = Path("/home/bangtliu/coord/logs")


@g.jit
def run(
    Reference, A, Flags, Epoch, Out, Times, MODE: gl.constexpr, CONSUMERS: gl.constexpr
):
    pid = gl.program_id(0)
    tag = gl.load(Epoch)
    if pid == 0:
        i = gl.arange(0, 16384, layout=gl.BlockedLayout([4], [32], [4], [0]))
        store_tile_words(
            A.to(gl.pointer_type(gl.int32)) + (i // 2048) * 8 * 2048 + i % 2048,
            gl.load(Reference.to(gl.pointer_type(gl.int32)) + i),
            i >= 0,
            16384,
        )
        scoped.publish_flags(Flags, tag, CONSUMERS)
    seen, polls = scoped.wait_flag(
        Flags + pid * 32, tag, 10_000_000, MAX_SPINS=1_048_576
    )
    start = sync.clock()
    if MODE == 1:
        gl.inline_asm_elementwise(
            "s_wait_loadcnt 0\nglobal_inv scope:SCOPE_DEV\nv_mov_b32 $0,0",
            "=v,~{memory}",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    r = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [16, 128], [1, 0]
    )
    sa = gl.allocate_shared_memory(gl.bfloat16, [2, 16, 128], shared)
    acc = gl.full((), 0, gl.float32, gl.BlockedLayout([1], [32], [4], [0]))
    for step in range(32):
        ptr = A + r[:, None] * (8 * 4096) + step * 128 + k[None, :]
        mask = (r[:, None] < 8).broadcast_to((16, 128))
        if MODE == 0:
            sa.index(step % 2).store(load_bf16_tile(ptr, mask, 16, 128))
        else:
            gfx1250.async_copy.global_to_shared(
                sa.index(step % 2),
                ptr,
                mask,
                0,
                cache_modifier=".cg" if MODE == 2 else "",
            )
            gfx1250.async_copy.commit_group()
            gfx1250.async_copy.wait_group(0)
        gl.barrier()
        value = sa.index(step % 2).load(layout).to(gl.float32)
        acc += gl.sum(gl.sum(value, 1), 0)
        gl.barrier()
    end = sync.clock()
    gl.store(Out + pid, gl.where(seen == tag, acc, float("nan")))
    gl.store(Times + pid * 16, end - start)
    gl.store(Times + pid * 16 + 1, polls)


def main():
    require_gpu0()
    torch.manual_seed(7)
    ref = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    data = torch.empty((64, 4096), device="cuda", dtype=torch.bfloat16)
    flags = torch.zeros((128, 32), device="cuda", dtype=torch.int32)
    epoch = torch.ones((), device="cuda", dtype=torch.int32)
    out = torch.empty(128, device="cuda")
    times = torch.zeros((128, 16), device="cuda", dtype=torch.int64)
    rows = []
    for consumers in (64, 128):
        for mode in (0, 1, 2):

            def call():
                epoch.add_(1)
                return run[(consumers,)](
                    ref,
                    data,
                    flags,
                    epoch,
                    out,
                    times,
                    mode,
                    consumers,
                    num_warps=4,
                    enable_fp_fusion=False,
                )

            kernel = call()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                out[:consumers].double(),
                ref.double().sum().expand(consumers),
                atol=0.05,
                rtol=1e-5,
            )
            timing = graph_times(call, samples=30)
            # New data in the same allocation must become visible on the next tag.
            ref.neg_()
            call()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                out[:consumers].double(),
                ref.double().sum().expand(consumers),
                atol=0.05,
                rtol=1e-5,
            )
            ticks = times[:consumers, 0].double().cpu() / 100
            row = dict(
                mode=mode,
                consumers=consumers,
                graph=timing,
                read_us=dict(
                    min=float(ticks.min()),
                    median=float(ticks.median()),
                    max=float(ticks.max()),
                ),
                vgprs=kernel.n_regs,
                lds=kernel.metadata.shared,
                spills=kernel.n_spills,
                correct=True,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            for fmt in ("amdgcn", "ttgir"):
                (ROOT / f"phase40c-Aonly-m{mode}-c{consumers}.{fmt}").write_text(
                    kernel.asm[fmt]
                )
    (ROOT / "phase40c-Aonly.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""GPU0 barrier and virtual-peer one-shot timings at the validated expert grid."""

import argparse
import json
from pathlib import Path

import torch
from bench_collective_fast import run_case
from bench_hbm import graph_times
from layer_fixture import require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped, sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)

ROOT = Path("/home/bangtliu/coord/logs")


@g.jit
def barriers(Storage, Epoch, Ready, Times, GRID: gl.constexpr, REPEAT: gl.constexpr):
    pid = gl.program_id(0)
    epoch = gl.load(Epoch)
    tag = (epoch - 1) * REPEAT
    ready = gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    before = sync.clock()
    for i in range(REPEAT):
        ok, _, _ = scoped.counter_barrier(Storage, tag + i + 1, 100_000, GRID, 12)
        ready &= ok
    after = sync.clock()
    gl.store(Ready + pid, ready)
    gl.store(Times + pid, after - before)


@g.jit
def advance(Epoch):
    gl.store(Epoch, gl.load(Epoch) + 1)


def main():
    require_gpu0()
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, choices=(4096, 8192))
    parser.add_argument("--total-only", action="store_true")
    args = parser.parse_args()
    proof = json.loads((ROOT / "phase3e1f1-real-residency.json").read_text())
    grid = proof["grid"]
    assert proof["ready_count"] == grid and proof["payload_good_count"] == grid
    if args.width:
        run_case(
            4,
            args.width,
            grid // 4,
            "oneshot",
            not args.total_only,
            waves=4,
            grid_limit=grid,
            log_prefix="phase3e1f1",
        )
        return
    storage = torch.zeros(
        (grid // 12 + 1 + grid) * 32, device="cuda", dtype=torch.int32
    )
    epoch = torch.ones(1, device="cuda", dtype=torch.int32)
    ready = torch.zeros(grid, device="cuda", dtype=torch.int32)
    times = torch.empty(grid, device="cuda", dtype=torch.int64)
    k = barriers.warmup(
        storage, epoch, ready, times, GRID=grid, REPEAT=100, num_warps=4, grid=(grid,)
    )
    occupancy = kernel_occupancy("isolated-two-level-barrier", k)
    assert occupancy.active_ctas_per_cu >= proof["ctas_per_cu"]

    def call():
        barriers[(grid,)](
            storage, epoch, ready, times, GRID=grid, REPEAT=100, num_warps=4
        )
        advance[(1,)](epoch)

    call()
    torch.cuda.synchronize()
    assert ready.eq(1).all()
    total = graph_times(call, samples=30)
    assert ready.eq(1).all()
    result = dict(
        grid=grid,
        waves=4,
        group=12,
        repeats_per_launch=100,
        ready_count=int(ready.sum()),
        total=total,
        barrier_median_us=float(times.double().median() / 100 / 100),
        barrier_max_us=float(times.double().max() / 100 / 100),
        registers=k.n_regs,
        spills=k.n_spills,
        shared=k.metadata.shared,
    )
    (ROOT / "phase3e1f1-isolated-barrier.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (ROOT / "phase3e1f1-isolated-barrier.amdgcn").write_text(k.asm["amdgcn"])
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

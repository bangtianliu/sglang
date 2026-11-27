"""gfx1250 synchronization components; GPU0 under timeout -s KILL 300."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped, sync

LOG = Path("/home/bangtliu/coord/logs")


@gluon.jit
def advance(Epoch):
    gl.store(Epoch, gl.load(Epoch) + 1)


@gluon.jit
def probe(
    Storage,
    Epoch,
    Stats,
    Out,
    Enabled,
    KIND: gl.constexpr,
    N: gl.constexpr,
    GRID: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
    i = gl.arange(0, 16384, layout=layout)
    shared = gl.allocate_shared_memory(
        gl.int32, [16384], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    pid = gl.program_id(0)
    tag = gl.load(Epoch)
    enabled = gl.load(Enabled)
    shared.store(i + pid)
    gl.barrier()
    start = sync.clock()
    cycles_start = sync.shader_cycles()
    value = gl.full((), 0, gl.uint64, layout)
    if enabled:
        for _ in range(N):
            if KIND == "realtime":
                value = sync.clock()
            elif KIND == "cycles":
                value = sync.shader_cycles()
            elif KIND == "shared_load":
                value = scoped.load_words(Storage).to(gl.uint64)
            elif KIND == "own_load":
                value = scoped.load_words(Storage + pid * 32).to(gl.uint64)
            elif KIND == "shared_flag":
                value = scoped.load_flag(Storage).to(gl.uint64)
            elif KIND == "own_flag":
                value = scoped.load_flag(Storage + pid * 32).to(gl.uint64)
            elif KIND == "atomic":
                value = gl.atomic_add(Storage, 1, sem="relaxed", scope="gpu").to(
                    gl.uint64
                )
            elif KIND == "join":
                # One-lane value followed by cross-wave reduction and CTA join,
                # matching load_flag's broadcast without its global load.
                lanes = gl.arange(0, 256, layout=layout)
                v = gl.where(lanes == 0, (value + tag).to(gl.int32), 0)
                value = gl.sum(v, 0).to(gl.uint64)
                gl.barrier()
    cycles_end = sync.shader_cycles()
    end = sync.clock()
    live = shared.load(layout)
    good = gl.sum((live == i + pid).to(gl.int32), 0) == 16384
    offset = ((tag % 1000) * GRID + pid) * 3
    gl.store(Stats + offset, end - start)
    gl.store(Stats + offset + 1, cycles_end - cycles_start)
    gl.store(Stats + offset + 2, value)
    gl.store(Out + pid, good)


def run(kind, grid, n):
    storage = torch.zeros(grid * 32, device="cuda", dtype=torch.int32)
    epoch = torch.ones(1, device="cuda", dtype=torch.int32)
    stats = torch.zeros((1000, grid, 3), device="cuda", dtype=torch.int64)
    out = torch.zeros(grid, device="cuda", dtype=torch.int32)
    enabled = torch.zeros(1, device="cuda", dtype=torch.int32)
    kwargs = dict(KIND=kind, N=n, GRID=grid, num_warps=8)
    kernel = probe[(grid,)](storage, epoch, stats, out, enabled, **kwargs)
    torch.cuda.synchronize()
    asm = kernel.asm["amdgcn"]
    assert "global_wb" not in asm and "global_inv" not in asm and "SCOPE_SYS" not in asm
    (LOG / f"phase2.2f-{kind}-grid{grid}-N{n}.amdgcn").write_text(asm)
    row = dict(
        kind=kind,
        grid=grid,
        operations_per_cta=n,
        warps=8,
        registers=kernel.n_regs,
        shared_bytes=kernel.metadata.shared,
    )
    for mode in (0, 1):
        enabled.fill_(mode)
        storage.zero_()
        epoch.fill_(1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(1000):
                probe[(grid,)](storage, epoch, stats, out, enabled, **kwargs)
                advance[(1,)](epoch)
        samples = []
        for _ in range(3):
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / 1000)
        assert out.all().item()
        data = stats.cpu().numpy()
        times = data[:, :, 0] / 100
        cycles = data[:, :, 1]
        row["work" if mode else "baseline"] = dict(
            median_us=float(np.median(times)),
            max_us=float(times.max()),
            host_us=statistics.median(samples),
            host_samples_us=samples,
            median_cycles=float(np.median(cycles)),
            max_cycles=int(cycles.max()),
        )
        if mode and kind == "atomic":
            assert storage[0].item() == 3000 * grid * n
    row["net_us_per_operation"] = (
        row["work"]["median_us"] - row["baseline"]["median_us"]
    ) / n
    row["net_host_us_per_operation"] = (
        row["work"]["host_us"] - row["baseline"]["host_us"]
    ) / n
    print(json.dumps(row), flush=True)
    return row


def main():
    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.startswith("gfx1250")
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrival-only", action="store_true")
    args = parser.parse_args()
    if args.arrival_only:
        row = run("atomic", props.multi_processor_count, 1)
        (LOG / "phase2.2f-arrival-N1.json").write_text(json.dumps(row, indent=2))
        return
    rows = []
    for kind, grid in [
        ("realtime", 1),
        ("realtime", props.multi_processor_count),
        ("cycles", 1),
        ("cycles", props.multi_processor_count),
        ("shared_load", props.multi_processor_count),
        ("own_load", props.multi_processor_count),
        ("shared_flag", props.multi_processor_count),
        ("own_flag", props.multi_processor_count),
        ("atomic", props.multi_processor_count),
        ("join", props.multi_processor_count),
    ]:
        rows.append(run(kind, grid, 32))
        (LOG / "phase2.2f-isolation.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

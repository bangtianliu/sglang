"""GPU0-only barrier cost breakdown; run under timeout -s KILL 300."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped, sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.barrier import (
    arrive_and_wait,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.counter_barrier import (
    counter_barrier,
)


@gluon.jit
def advance(Epoch):
    gl.store(Epoch, gl.load(Epoch) + 1)


@gluon.jit
def probe(
    Storage,
    Epoch,
    Out,
    Stats,
    Enabled,
    GRID: gl.constexpr,
    ALGO: gl.constexpr,
    GROUP: gl.constexpr,
    SLEEP_EVERY: gl.constexpr,
    POLL_SEM: gl.constexpr,
    CLOCK_EVERY: gl.constexpr,
    SPREAD_FLAGS: gl.constexpr,
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
    before = sync.clock()
    ready = True
    polls = 0
    sleeps = 0
    if enabled:
        if ALGO == "scoped":
            ready, polls, sleeps = scoped.arrive_and_wait(Storage, tag, 100_000, GRID)
        elif ALGO == "scoped-counter":
            ready, polls, sleeps = scoped.counter_barrier(
                Storage,
                tag,
                100_000,
                GRID,
                GROUP,
                CLOCK_EVERY=CLOCK_EVERY,
                SPREAD_FLAGS=SPREAD_FLAGS,
            )
        elif ALGO == "vector":
            ready, polls, sleeps = arrive_and_wait(
                Storage,
                tag,
                10_000_000,
                GRID,
                SLEEP_EVERY=SLEEP_EVERY,
                POLL_SEM=POLL_SEM,
                RETURN_STATS=True,
            )
        else:
            ready, polls, sleeps = counter_barrier(
                Storage,
                tag,
                10_000_000,
                GRID,
                GROUP,
                SLEEP_EVERY=SLEEP_EVERY,
                POLL_SEM=POLL_SEM,
            )
    after = sync.clock()
    values = shared.load(layout)
    good = gl.sum((values == i + pid).to(gl.int32), 0) == 16384
    gl.store(Out + pid, ready & good)
    offset = ((tag % 1000) * GRID + pid) * 4
    gl.store(Stats + offset, after - before)
    gl.store(Stats + offset + 1, polls)
    gl.store(Stats + offset + 2, sleeps)
    gl.store(Stats + offset + 3, ready & good)


def run_case(
    *,
    name,
    algo,
    group,
    sleep_every,
    sem,
    grid,
    clock_every=1,
    spread=False,
    phase=None,
):
    storage = torch.zeros(
        max(
            1 << (grid - 1).bit_length(),
            (grid // group + 2 + (grid if spread else 0)) * 32,
        ),
        device="cuda",
        dtype=torch.int32,
    )
    epoch = torch.ones(1, device="cuda", dtype=torch.int32)
    out = torch.zeros(grid, device="cuda", dtype=torch.int32)
    stats = torch.zeros((1000, grid, 4), device="cuda", dtype=torch.int64)
    enabled = torch.zeros(1, device="cuda", dtype=torch.int32)
    kwargs = dict(
        GRID=grid,
        ALGO=algo,
        GROUP=group,
        SLEEP_EVERY=sleep_every,
        POLL_SEM=sem,
        CLOCK_EVERY=clock_every,
        SPREAD_FLAGS=spread,
        num_warps=8,
    )
    kernel = probe[(grid,)](storage, epoch, out, stats, enabled, **kwargs)
    advance[(1,)](epoch)
    torch.cuda.synchronize()
    asm = kernel.asm["amdgcn"]
    assert "SCOPE_SYS" not in asm
    if algo.startswith("scoped"):
        assert "global_wb" not in asm and "global_inv" not in asm
    if phase is None:
        phase = "2.2d" if algo.startswith("scoped") else "2.2b"
    Path(f"/home/bangtliu/coord/logs/phase{phase}-{name}.amdgcn").write_text(asm)
    results = dict(
        name=name,
        registers=kernel.n_regs,
        shared_bytes=kernel.metadata.shared,
        scope="gpu",
        clock_every=clock_every,
        spread_flags=spread,
    )
    for mode in (0, 1):
        enabled.fill_(mode)
        storage.zero_()
        epoch.fill_(1)
        stats.zero_()
        probe[(grid,)](storage, epoch, out, stats, enabled, **kwargs)
        advance[(1,)](epoch)
        torch.cuda.synchronize()
        assert out.all().item(), (name, mode, "preflight")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(1000):
                probe[(grid,)](storage, epoch, out, stats, enabled, **kwargs)
                advance[(1,)](epoch)
        samples = []
        for _ in range(3):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / 1000)
        assert out.all().item(), (name, mode, out.sum().item())
        data = stats.cpu().numpy()
        assert data[:, :, 3].all(), (name, mode, "per-launch validation")
        clocks = data[:, :, 0] / 100
        label = "barrier" if mode else "baseline"
        results[label] = dict(
            host_graph_us=statistics.median(samples),
            host_samples_us=samples,
            median_us=float(np.median(clocks)),
            max_us=float(clocks.max()),
            p99_us=float(np.percentile(clocks, 99)),
            median_cta_us=np.median(clocks, axis=0).tolist(),
            mean_polls=float(data[:, :, 1].mean()),
            max_polls=int(data[:, :, 1].max()),
            mean_sleeps=float(data[:, :, 2].mean()),
            max_sleeps=int(data[:, :, 2].max()),
        )
    results["host_difference_us"] = (
        results["barrier"]["host_graph_us"] - results["baseline"]["host_graph_us"]
    )
    summary = {k: v for k, v in results.items() if k not in ("baseline", "barrier")}
    for label in ("baseline", "barrier"):
        summary[label] = {
            k: v for k, v in results[label].items() if k != "median_cta_us"
        }
    print(json.dumps(summary), flush=True)
    return results


def main():
    assert torch.cuda.device_count() == 1, "expose GPU0 only"
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.startswith("gfx1250")
    grid = props.multi_processor_count
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", action="store_true")
    parser.add_argument("--scoped", action="store_true")
    parser.add_argument("--cadence", action="store_true")
    args = parser.parse_args()
    if args.cadence:
        rows = []
        output = Path("/home/bangtliu/coord/logs/phase2.2g-barrier.json")
        for group, spread, every in (
            (grid, False, 1),
            (grid, False, 64),
            (grid, True, 1),
            (grid, True, 64),
            (12, True, 64),
        ):
            name = f"group{group}-{'spread' if spread else 'shared'}-clock{every}"
            rows.append(
                run_case(
                    name=name,
                    algo="scoped-counter",
                    group=group,
                    sleep_every=0,
                    sem="unused",
                    grid=grid,
                    clock_every=every,
                    spread=spread,
                    phase="2.2g",
                )
            )
            output.write_text(json.dumps(rows, indent=2))
        return
    cases = [("vector-acquire-sleep1", "vector", grid, 1, "acquire")]
    for group in (grid, 12):
        prefix = "single" if group == grid else "tree12"
        cases.append((f"{prefix}-acquire-no-sleep", "counter", group, 0, "acquire"))
        for every in (0, 8, 32):
            cases.append(
                (f"{prefix}-relaxed-sleep{every}", "counter", group, every, "relaxed")
            )
    if args.extra:
        cases = [
            (f"vector-acquire-sleep{n}", "vector", grid, n, "acquire")
            for n in (0, 8, 32)
        ]
    if args.scoped:
        cases = [
            ("scoped", "scoped", grid, 0, "unused"),
            ("scoped-single", "scoped-counter", grid, 0, "unused"),
            ("scoped-tree12", "scoped-counter", 12, 0, "unused"),
        ]
    rows = []
    output = Path("/home/bangtliu/coord/logs/phase2.2b-breakdown.json")
    if args.extra:
        output = output.with_name("phase2.2b-extra.json")
    if args.scoped:
        output = output.with_name("phase2.2d-barrier.json")
    for name, algo, group, every, sem in cases:
        rows.append(
            run_case(
                name=name, algo=algo, group=group, sleep_every=every, sem=sem, grid=grid
            )
        )
        output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

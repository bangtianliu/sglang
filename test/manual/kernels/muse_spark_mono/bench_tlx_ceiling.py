"""GPU0-only TLX stream ceiling and bounded dependent-read latency.

Diagnostic kernels only; no production stage or peer allocation changes.
"""

import argparse
import functools
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx
from layer_fixture import require_gpu0
from paired45 import measure
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW4,
)

LOG = Path("/home/bangtliu/coord/logs")


@triton.jit
def stream(X, Y, N: tl.constexpr, GRID: tl.constexpr, CACHE: tl.constexpr):
    pid = tl.program_id(0)
    i = tlx.require_layout(tl.arange(0, 4096), ROW4)
    total = tl.full((4096,), 0, tl.float32)
    for block in range(pid, tl.cdiv(N, 4096), GRID):
        at = block * 4096 + i
        total += tl.load(X + at, at < N, 0, cache_modifier=CACHE)
    tl.store(Y + pid, tl.sum(total, 0))


@triton.jit
def chase(X, Y, STEPS: tl.constexpr, CACHE: tl.constexpr):
    # One active lane follows one cycle. Every offset comes from the last load;
    # fixed iteration count bounds execution independently of memory contents.
    i = tl.arange(0, 128)
    at = tl.full((128,), 0, tl.int32)
    for _ in range(STEPS):
        at = tl.load(X + at * 32, i == 0, 0, cache_modifier=CACHE, volatile=True)
    tl.store(Y, tl.sum(at, 0))


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--bytes", type=int, default=1 << 30)
    p.add_argument("--latency", action="store_true")
    a = p.parse_args()
    n = a.bytes // 4
    xs = [torch.ones((n,), device="cuda") for _ in range(3)]
    rows = []
    for grid in (192, 768, 1728):
        cases = {}
        for name, g in [("wide", triton.cdiv(n, 4096)), ("resident", grid)]:
            ys = [torch.empty((g,), device="cuda") for _ in xs]
            pid = torch.arange(g, device="cuda", dtype=torch.int64)
            count = torch.clamp((triton.cdiv(n, 4096) - 1 - pid) // g + 1, min=0)
            expected = count.float() * 4096

            def call(g=g, ys=ys):
                for x, y in zip(xs, ys):
                    stream[(g,)](x, y, n, g, "", num_warps=4)

            def check(ys=ys, expected=expected):
                assert all(torch.equal(y, expected) for y in ys), "stream checksum"

            cases[name] = (lambda: None, call, check)
        result, _ = measure(cases)
        for label, r in result.items():
            r["per_copy_us"] = r["median_us"] / 3
            r["logical_TB_s"] = a.bytes / r["per_copy_us"] / 1e6
        rows.append(dict(grid=grid, results=result))
        print(
            "stream",
            a.bytes,
            grid,
            {k: r["logical_TB_s"] for k, r in result.items()},
            flush=True,
        )
    compiled = stream.warmup(xs[0], ys[0], n, grid, "", num_warps=4, grid=(grid,))
    dump(compiled, "T9-ceiling/stream", "tlx")
    latency = []
    if a.latency:
        generator = torch.Generator().manual_seed(20261118)
        for size in (32768, 4 << 20, 128 << 20, 1 << 30):
            lines = size // 128
            order = torch.randperm(lines, generator=generator)
            nxt = torch.empty((lines,), dtype=torch.int32)
            nxt[order] = order.roll(-1).to(torch.int32)
            x = torch.zeros((lines * 32,), device="cuda", dtype=torch.int32)
            x[::32] = nxt.to("cuda")
            y = torch.zeros((), device="cuda", dtype=torch.int32)
            expected = 0
            for _ in range(2048):
                expected = int(nxt[expected])

            # Evict prior pointer-chain cache lines before every timed replay.
            def flush():
                stream[(768,)](xs[0], flush_y, n, 768, "", num_warps=4)

            flush_y = torch.empty((768,), device="cuda")
            cases = {}
            for name, cache in [("default", ""), ("DEV", ".cg")]:
                cases[name] = (
                    flush,
                    functools.partial(chase[(1,)], x, y, 2048, cache, num_warps=4),
                    lambda: torch.testing.assert_close(
                        y,
                        torch.tensor(expected, device="cuda", dtype=torch.int32),
                        rtol=0,
                        atol=0,
                    ),
                )
            result, _ = measure(cases)
            for r in result.values():
                r["event_ns_per_dependent_load"] = r["median_us"] * 1000 / 2048
            latency.append(
                dict(
                    working_set_bytes=size,
                    steps=2048,
                    eviction_bytes=a.bytes,
                    results=result,
                )
            )
            for label, cache in [("default", ""), ("DEV", ".cg")]:
                compiled = chase.warmup(x, y, 2048, cache, num_warps=4, grid=(1,))
                dump(compiled, "T9-ceiling/chase-" + str(size), label)
            print(
                "chase",
                size,
                {k: r["event_ns_per_dependent_load"] for k, r in result.items()},
                flush=True,
            )
    (LOG / f"phaseT9-ceiling-{a.bytes}.json").write_text(
        json.dumps(
            dict(
                bytes=a.bytes,
                copies=3,
                stream=rows,
                latency=latency,
                limit="logical read throughput; dependent latency includes small graph/kernel overhead; one active lane; eviction before each replay; no hardware counter",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

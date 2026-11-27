"""Bounded one-GPU residency probe with eight warps and64KiB live LDS."""

import json
import unittest
from pathlib import Path

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.barrier import (
    arrive_and_wait,
)
from sglang.test.test_utils import CustomTestCase


@gluon.jit
def barrier_kernel(
    Flags,
    Counter,
    Out,
    Timings,
    GRID: gl.constexpr,
    OMIT_LAST: gl.constexpr = False,
    TIMEOUT_TICKS: gl.constexpr = 100_000_000,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [8], [0])
    i = gl.arange(0, 16384, layout=layout)
    shared = gl.allocate_shared_memory(
        gl.int32, [16384], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    pid = gl.program_id(0)
    tag = gl.load(Counter)
    shared.store(i + pid)
    before = sync.clock()
    ready = False
    if not OMIT_LAST or pid < GRID - 1:
        ready = arrive_and_wait(Flags, tag, TIMEOUT_TICKS, GRID)
    after = sync.clock()
    values = shared.load(layout)
    good = gl.sum((values == i + pid).to(gl.int32), 0) == 16384
    gl.store(Out + pid, ready & good)
    if pid == 0:
        gl.store(Timings + tag % 1000, after - before)
        gl.store(Counter, tag + 1)


class TestGridBarrier(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
            0
        ).gcnArchName.startswith("gfx1250"):
            self.skipTest("requires gfx1250")
        assert torch.cuda.device_count() == 1, "expose GPU0 only"

    def test_missing_arrival_times_out(self):
        grid = torch.cuda.get_device_properties(0).multi_processor_count
        flags = torch.zeros(grid, device="cuda", dtype=torch.int32)
        flags[-1] = 99
        counter = torch.ones(1, device="cuda", dtype=torch.int32)
        out = torch.ones(grid, device="cuda", dtype=torch.int32)
        timings = torch.zeros(1000, device="cuda", dtype=torch.int64)
        barrier_kernel[(grid,)](
            flags,
            counter,
            out,
            timings,
            GRID=grid,
            OMIT_LAST=True,
            TIMEOUT_TICKS=100_000,
            num_warps=8,
        )
        torch.cuda.synchronize()
        assert not out.any().item()
        elapsed_us = timings[1].item() / 100
        assert 1000 <= elapsed_us < 100_000, elapsed_us
        print(
            f"PASS missing arrival and stale tag rejected; deadline1000us actual={elapsed_us:.2f}us",
            flush=True,
        )

    def test_residency(self):
        units = torch.cuda.get_device_properties(0).multi_processor_count
        rows = []
        for grid in (units, units + 1):
            flags = torch.zeros(grid, device="cuda", dtype=torch.int32)
            counter = torch.ones(1, device="cuda", dtype=torch.int32)
            out = torch.zeros(grid, device="cuda", dtype=torch.int32)
            timings = torch.zeros(1000, device="cuda", dtype=torch.int64)
            kernel = barrier_kernel[(grid,)](
                flags, counter, out, timings, GRID=grid, num_warps=8
            )
            torch.cuda.synchronize()
            success = bool(out.all().item())
            Path(f"/home/bangtliu/coord/logs/phase2.2-grid{grid}.amdgcn").write_text(
                kernel.asm["amdgcn"]
            )
            row = dict(
                grid=grid,
                success=success,
                passed_ctas=int(out.sum().item()),
                registers=kernel.n_regs,
                shared_bytes=kernel.metadata.shared,
            )
            if grid == units:
                assert success, row
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    barrier_kernel[(grid,)](
                        flags, counter, out, timings, GRID=grid, num_warps=8
                    )
                for _ in range(1000):
                    graph.replay()
                torch.cuda.synchronize()
                assert out.all().item()
                row["median_barrier_us_1000"] = timings.float().median().item() / 100
            rows.append(row)
            print(json.dumps(row), flush=True)
        Path("/home/bangtliu/coord/logs/phase2.2-barrier-results.json").write_text(
            json.dumps(rows, indent=2)
        )


if __name__ == "__main__":
    unittest.main()

"""Agent-scoped gfx1250 mailbox probes; GPU0 under timeout -s KILL 300."""

import unittest
from pathlib import Path

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
    sync,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    vector_memory as vm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective_fast import (
    wait_sources,
)
from sglang.test.test_utils import CustomTestCase

LOG = Path("/home/bangtliu/coord/logs")


@gluon.jit
def ordering(Payload, Flag, Results, Tag):
    i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [8], [0]))
    if gl.program_id(0) == 0:
        scoped.store_words(Payload + i, i + Tag, gl.full_like(i, True, dtype=gl.int1))
        scoped.publish(Flag, Tag)
    else:
        seen, spins = scoped.wait_flag(Flag, Tag, 100_000)
        values = scoped.load_words(Payload + i)
        good = gl.sum((values == i + Tag).to(gl.int32), 0) == 1024
        gl.store(Results + Tag - 1, good & (seen == Tag))


@gluon.jit
def missing(
    Flags,
    Out,
    Times,
    Polls,
    GRID: gl.constexpr,
    CAP: gl.constexpr,
    GROUP: gl.constexpr,
    SPREAD: gl.constexpr = False,
):
    pid = gl.program_id(0)
    start = sync.clock()
    ready = False
    polls = 0
    deadline: gl.constexpr = 10_000_000 if CAP == 16 else 100_000
    if pid < GRID - 1:
        if GROUP == 0:
            ready, polls, sleeps = scoped.arrive_and_wait(
                Flags, 1, deadline, GRID, MAX_SPINS=CAP
            )
        else:
            ready, polls, sleeps = scoped.counter_barrier(
                Flags, 1, deadline, GRID, GROUP, MAX_SPINS=CAP, SPREAD_FLAGS=SPREAD
            )
    end = sync.clock()
    gl.store(Out + pid, ready)
    gl.store(Times + pid, end - start)
    gl.store(Polls + pid, polls)


@gluon.jit
def spread_payload(Payload, Storage, Epoch, Results, GRID: gl.constexpr):
    pid = gl.program_id(0)
    tag = gl.load(Epoch)
    i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [8], [0]))
    scoped.store_words(
        Payload + pid * 1024 + i, i + tag + pid, gl.full_like(i, True, dtype=gl.int1)
    )
    ready, polls, sleeps = scoped.counter_barrier(
        Storage, tag, 100_000, GRID, GRID, SPREAD_FLAGS=True
    )
    neighbor = (pid + 1) % GRID
    values = scoped.load_words(Payload + neighbor * 1024 + i)
    valid = gl.sum((values == i + tag + neighbor).to(gl.int32), 0) == 1024
    gl.store(Results + (tag - 1) * GRID + pid, ready & valid)


@gluon.jit
def advance_epoch(Epoch):
    gl.store(Epoch, gl.load(Epoch) + 1)


@gluon.jit
def vector_copy(A, B, WORLD: gl.constexpr, Q: gl.constexpr, PACK: gl.constexpr):
    i = 4 * gl.arange(0, Q, layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]))
    words = vm.load_sources(A + i, i < 4 * Q - 4, 4 * Q, WORLD, PACK)
    pointers = ()
    values = ()
    for src in gl.static_range(WORLD):
        pointers += (B + src * 4 * Q + i,)
        values += (
            words[src * 4],
            words[src * 4 + 1],
            words[src * 4 + 2],
            words[src * 4 + 3],
        )
    vm.store_destinations(pointers, values, i < 4 * Q - 4, WORLD, PACK)
    gl.barrier()


@gluon.jit
def parallel_flags(Flags, Out, Times, WORLD: gl.constexpr, BPR: gl.constexpr):
    pid = gl.program_id(0)
    stride: gl.constexpr = WORLD * BPR * 32
    start = sync.clock()
    ready = wait_sources(
        Flags, 1, 0, pid // BPR, pid % BPR, (0, stride, WORLD * stride), WORLD, BPR
    )
    end = sync.clock()
    gl.store(Out + pid, ready)
    gl.store(Times + pid, end - start)


def evidence(name, kernel):
    asm = kernel.asm["amdgcn"]
    assert "global_wb" not in asm and "global_inv" not in asm
    assert "SCOPE_SYS" not in asm
    (LOG / f"phase2.2d-{name}.amdgcn").write_text(asm)


class TestScoped(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
            0
        ).gcnArchName.startswith("gfx1250"):
            self.skipTest("requires gfx1250")
        assert torch.cuda.device_count() == 1, "expose GPU0 only"

    def test_payload_ordering_1001_launches(self):
        payload = torch.zeros(1024, device="cuda", dtype=torch.int32)
        flag = torch.zeros(1, device="cuda", dtype=torch.int32)
        results = torch.zeros(1001, device="cuda", dtype=torch.int32)
        kernel = ordering[(2,)](payload, flag, results, 1, num_warps=8)
        evidence("scoped-ordering", kernel)
        for tag in range(2, 1002):
            ordering[(2,)](payload, flag, results, tag, num_warps=8)
        assert results.all().item(), int(results.sum().item())
        print("PASS 4KiB scoped payload, all 1001 launches verified", flush=True)

    def test_missing_arrival_and_iteration_cap(self):
        grid = torch.cuda.get_device_properties(0).multi_processor_count
        for cap, group in ((c, g) for g in (0, grid, 12) for c in (16, 4096)):
            flags = torch.zeros(
                max(1 << (grid - 1).bit_length(), (grid // 12 + 2) * 32),
                device="cuda",
                dtype=torch.int32,
            )
            flags[grid - 1 if group == 0 else (grid // group + 1) * 32] = 99
            out = torch.ones(grid, device="cuda", dtype=torch.int32)
            times = torch.zeros(grid, device="cuda", dtype=torch.int64)
            polls = torch.zeros(grid, device="cuda", dtype=torch.int32)
            kernel = missing[(grid,)](
                flags, out, times, polls, GRID=grid, CAP=cap, GROUP=group, num_warps=8
            )
            assert not out.any().item()
            assert int(polls.max().item()) <= cap
            us = times[:-1].double() / 100
            assert float(us.max()) < 10_000, float(us.max())
            if cap == 16:
                assert int(polls[:-1].min().item()) == cap
            else:
                assert float(us.min()) >= 1000
            evidence(f"scoped-missing-group{group}-cap{cap}", kernel)
            print(
                f"PASS missing/stale flag group={group} cap={cap} median_us={float(us.median()):.2f} max_us={float(us.max()):.2f} max_polls={int(polls.max())}",
                flush=True,
            )

    def test_spread_completion_bounds(self):
        grid = torch.cuda.get_device_properties(0).multi_processor_count
        for group in (grid, 12):
            base = (grid // group + 1) * 32
            for cap in (16, 65536):
                flags = torch.zeros(base + grid * 32, device="cuda", dtype=torch.int32)
                flags[base::32] = 99
                out = torch.ones(grid, device="cuda", dtype=torch.int32)
                times = torch.zeros(grid, device="cuda", dtype=torch.int64)
                polls = torch.zeros(grid, device="cuda", dtype=torch.int32)
                kernel = missing[(grid,)](
                    flags,
                    out,
                    times,
                    polls,
                    GRID=grid,
                    CAP=cap,
                    GROUP=group,
                    SPREAD=True,
                    num_warps=8,
                )
                assert not out.any().item()
                assert int(polls.max()) <= cap
                us = times[:-1].double() / 100
                assert float(us.max()) < 10000
                if cap == 16:
                    assert int(polls[:-1].min()) == cap
                else:
                    assert float(us.min()) >= 1000
                    assert int(polls[:-1].max()) < cap
                asm = kernel.asm["amdgcn"]
                assert (
                    "global_wb" not in asm
                    and "global_inv" not in asm
                    and "SCOPE_SYS" not in asm
                )
                (
                    LOG / f"phase2.2g-spread-bounds-group{group}-cap{cap}.amdgcn"
                ).write_text(asm)
                print(
                    f"PASS spread bounds group={group} cap={cap} max_us={float(us.max()):.2f} max_polls={int(polls.max())}",
                    flush=True,
                )

    def test_spread_grid_payload_1000_launches(self):
        grid = torch.cuda.get_device_properties(0).multi_processor_count
        payload = torch.zeros((grid, 1024), device="cuda", dtype=torch.int32)
        storage = torch.zeros((grid + 2) * 32, device="cuda", dtype=torch.int32)
        epoch = torch.ones(1, device="cuda", dtype=torch.int32)
        results = torch.zeros((1000, grid), device="cuda", dtype=torch.int32)
        kernel = spread_payload[(grid,)](
            payload, storage, epoch, results, GRID=grid, num_warps=8
        )
        advance_epoch[(1,)](epoch)
        for _ in range(999):
            spread_payload[(grid,)](
                payload, storage, epoch, results, GRID=grid, num_warps=8
            )
            advance_epoch[(1,)](epoch)
        assert results.all().item(), int(results.sum())
        asm = kernel.asm["amdgcn"]
        assert (
            "global_wb" not in asm
            and "global_inv" not in asm
            and "SCOPE_SYS" not in asm
        )
        (LOG / "phase2.2g-spread-payload.amdgcn").write_text(asm)
        print(f"PASS 1000 launches x {grid} CTAs x 4KiB neighbor payload", flush=True)

    def test_vector_batches_and_masks(self):
        for world, pack, warps in (
            (4, 1, 8),
            (4, 2, 8),
            (4, 4, 8),
            (8, 1, 8),
            (8, 2, 8),
            (8, 2, 16),
        ):
            q = warps * 32 * pack
            data = torch.arange(world * q * 4, device="cuda", dtype=torch.int32)
            out = torch.zeros_like(data)
            kernel = vector_copy[(1,)](
                data, out, WORLD=world, Q=q, PACK=pack, num_warps=warps
            )
            expected = data.reshape(world, -1).clone()
            expected[:, -4:] = 0
            assert torch.equal(out.reshape(world, -1), expected), (world, pack, warps)
            asm = kernel.asm["amdgcn"]
            assert "global_load_b128" in asm and "global_store_b128" in asm
            assert (
                "global_wb" not in asm
                and "global_inv" not in asm
                and "SCOPE_SYS" not in asm
            )
        print("PASS six masked b128 batching configurations", flush=True)

    def test_parallel_source_flag_timeout(self):
        for world in (4, 8):
            bpr = 16
            grid = world * bpr
            stride = world * bpr * 32
            flags = torch.ones(2 * world * stride, device="cuda", dtype=torch.int32)
            for rank in range(world):
                flags[
                    rank * stride + (world - 1) * bpr * 32 : rank * stride
                    + world * bpr * 32 : 32
                ] = 0
            out = torch.ones(grid, device="cuda", dtype=torch.int32)
            times = torch.zeros(grid, device="cuda", dtype=torch.int64)
            kernel = parallel_flags[(grid,)](
                flags, out, times, WORLD=world, BPR=bpr, num_warps=8
            )
            assert not out.any().item()
            assert float(times.max()) / 100 < 10000
            asm = kernel.asm["amdgcn"]
            (LOG / f"phase2.8-parallel-timeout-world{world}.amdgcn").write_text(asm)
            flags.fill_(1)
            parallel_flags[(grid,)](
                flags, out, times, WORLD=world, BPR=bpr, num_warps=8
            )
            assert out.all().item()
        print(
            "PASS world4/8 parallel missing-source and complete flag checks", flush=True
        )


if __name__ == "__main__":
    unittest.main()

"""Experimental GPU0 W13/W2 dataflow pair; not selected by the TLX chain.

Four-wave kernels retain distinct resource budgets. Each W2 task polls its own
128-byte line. W13 never waits. Payload stores and flag-side A reads are DEV
scoped. Callers must admit the combined resident CTA budget before dispatch.
"""

import msgspec
import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard as guard,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_memory as mem,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW4,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_core import (
    expert_tile,
)


@triton.jit
def initialize(Epoch, Counters, Record, Status):
    i = tl.arange(0, 2048)
    tl.store(Counters + i, 0, i < 64 * 32)
    tl.store(Status + i, 1, i < 1728)
    tl.store(Record + tl.arange(0, 8), 0)
    tl.atomic_add(Epoch, 1, sem="relaxed", scope="gpu")


@triton.jit
def produce(
    X,
    Sorted,
    Experts,
    NumPost,
    W,
    Scale,
    Out,
    Status,
    Failure,
    Epoch,
    Counters,
    Flags,
    GRID: tl.constexpr,
    SKIP: tl.constexpr = -1,
    Seed=None,
):
    pid = tl.program_id(0)
    tag = tl.load(Epoch)
    allowed = guard.enabled(Failure, 5)
    tasks = tl.load(NumPost) // 16 * 64
    if allowed:
        for task in range(pid, tasks, GRID):
            block = task // 64
            expert = tl.load(Experts + block)
            if pid != SKIP:
                guard.delay(Seed, task, 0)
                expert_tile(
                    X,
                    Sorted,
                    W,
                    Scale,
                    Out,
                    block,
                    expert,
                    task % 64,
                    True,
                    OUTPUT_AGENT=True,
                )
                mem.drain_stores()
                previous = tl.atomic_add(
                    Counters + block * 32, 1, sem="relaxed", scope="gpu"
                )
                if previous == 63:
                    consumers = block * 64 + tl.arange(0, 64)
                    mem.store_words(
                        Flags + consumers * 32,
                        tl.full((64,), tag, tl.int32),
                        tl.full((64,), True, tl.int1),
                    )
                    mem.drain_stores()
            else:
                tl.store(Status + pid, 0)


@triton.jit
def warm_weights(W, Scale, expert, tile):
    i = tlx.require_layout(tl.arange(0, 8192), ROW4)
    value = tl.load(W + expert * 4096 * 2048 + (tile * 64 + i // 128) * 2048 + i % 128)
    j = tlx.require_layout(tl.arange(0, 512), ROW4)
    scale = tl.load(Scale + expert * 4096 * 128 + (tile * 64 + j // 8) * 128 + j % 8)
    return tl.sum(value.to(tl.int32), 0) + tl.sum(scale.to(tl.int32), 0)


@triton.jit
def consume(
    X,
    Sorted,
    Experts,
    NumPost,
    W,
    Scale,
    Out,
    Status,
    Failure,
    Epoch,
    Flags,
    Record,
    WarmChecks,
    GRID: tl.constexpr,
    WARM: tl.constexpr = False,
    Seed=None,
):
    pid = tl.program_id(0)
    tag = tl.load(Epoch)
    tasks = tl.load(NumPost) // 16 * 64
    for task in range(pid, tasks, GRID):
        block = task // 64
        expert = tl.load(Experts + block)
        if WARM:
            checksum = warm_weights(W, Scale, expert, task % 64)
            tl.store(WarmChecks + task, checksum)
        # One line per consumer task. No shared hot polling cache line.
        flag = Flags + task * 32
        seen = mem.load_flag(flag)
        spins = 0
        while (seen != tag) & (spins < 4096):
            tl.inline_asm_elementwise(
                "s_sleep 1\nv_mov_b32 $0,0",
                "=v",
                [],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            seen = mem.load_flag(flag)
            spins += 1
        tl.store(WarmChecks + 4096 + task, spins)
        ok = seen == tag
        if ok:
            guard.delay(Seed, task, 1)
            expert_tile(
                X,
                Sorted,
                W,
                Scale,
                Out,
                block,
                expert,
                task % 64,
                False,
                CROSS_LAUNCH_A=False,
            )
        else:
            guard.finish(Failure, False, pid, 5)
            tl.store(Status + pid, 0)
            owner = tl.atomic_cas(Record, 0, 1, sem="relaxed", scope="gpu")
            if owner == 0:
                tl.store(Record + 1, pid)
                tl.store(Record + 2, task)
                tl.store(Record + 3, tag)
                tl.store(Record + 4, seen)
                tl.store(Record + 5, spins)


class Workspace(msgspec.Struct):
    epoch: object
    counters: object
    flags: object
    record: object
    checks: object
    stream: object


def make_workspace(device):
    return Workspace(
        torch.zeros((), device=device, dtype=torch.int32),
        torch.zeros((64 * 32,), device=device, dtype=torch.int32),
        torch.zeros((4096 * 32,), device=device, dtype=torch.int32),
        torch.zeros((8,), device=device, dtype=torch.int32),
        torch.zeros((8192,), device=device, dtype=torch.int32),
        torch.cuda.Stream(device=device),
    )

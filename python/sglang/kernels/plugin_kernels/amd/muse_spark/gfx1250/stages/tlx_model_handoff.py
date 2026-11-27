"""Experimental release/acquire S5 handoff; production does not select it.

Packed epoch/counters self-initialize; no per-call init launch is hidden. One
wave polls each private 128B flag line; other waves rendezvous once. An agent
acquire atomic load precedes ordinary cached activation loads.
"""

import msgspec
import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard as guard,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_core import (
    expert_tile,
)


@triton.jit
def wait_flag(Flag, Tag):
    lanes = tlx.require_layout(tl.arange(0, 128), ROW1, pin=True)
    saved, count, compare, seen = tl.inline_asm_elementwise(
        "v_mov_b32 $3,0\ns_mov_b32 $1,0\nv_cmp_ne_u32_e64 $0,0,$6\ns_and_saveexec_b32 $0,$0\ns_cbranch_execz 2f\n"
        "1:\nglobal_load_b32 $3,$4,off scope:SCOPE_DEV\ns_wait_loadcnt 0\nv_cmp_eq_u32_e64 $2,$3,$5\ns_cmp_lg_u32 $2,0\ns_cbranch_scc1 2f\n"
        "s_add_u32 $1,$1,1\ns_cmp_lt_u32 $1,4096\ns_cbranch_scc0 2f\ns_sleep 1\ns_branch 1b\n"
        "2:\ns_mov_b32 exec_lo,$0\ns_wait_xcnt 0x0",
        "=&s,=&s,=&s,=&v,v,v,v,~{scc}",
        [
            tlx.release_layout(Flag + tl.full((128,), 0, tl.int32)),
            tlx.release_layout(tl.full((128,), Tag, tl.int32)),
            tlx.release_layout((lanes == 0).to(tl.int32)),
        ],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    seen = tlx.require_layout(seen, ROW1, pin=True)
    count = tlx.require_layout(count, ROW1, pin=True)
    packed = tl.sum(
        tl.where(lanes == 0, ((seen == Tag).to(tl.int32)) | (count << 1), 0), 0
    )
    ok = (packed & 1) != 0
    polls = packed >> 1
    observed = Tag
    if ok:
        # LLVM acquire is compiler-visible and invalidates private caches once.
        # The tag cannot change again until this pair and its parent sum finish.
        acquired = tl.atomic_load(Flag, sem="acquire", scope="gpu")
        ok = acquired == Tag
        observed = acquired
    else:
        observed = tl.sum(seen, 0)
    return ok, polls, observed


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
    Epochs,
    Counters,
    Flags,
    GRID: tl.constexpr,
    SKIP: tl.constexpr = -1,
    Seed=None,
):
    pid = tl.program_id(0)
    tag = tl.load(Epochs) + 1
    allowed = guard.enabled(Failure, 5)
    ok = allowed & (pid != SKIP)
    if ok:
        tasks = tl.load(NumPost) // 16 * 64
        for task in range(pid, tasks, GRID):
            block = task // 64
            expert = tl.load(Experts + block)
            guard.delay(Seed, task, 0)
            expert_tile(X, Sorted, W, Scale, Out, block, expert, task % 64, True)
            # Global+local CTA rendezvous orders every producer wave before the
            # single logical counter RMW (unlike an LDS-only workgroup barrier).
            tl.debug_barrier()
            counter = Counters + block * 16
            epoch_bits = tag.to(tl.int64) << 32
            prior = tl.atomic_load(counter, sem="relaxed", scope="gpu")
            if (prior >> 32) != tag:
                tl.atomic_cas(counter, prior, epoch_bits, sem="relaxed", scope="gpu")
            previous = tl.atomic_add(counter, 1, sem="release", scope="gpu")
            if (previous & 0xFFFFFFFF) == 63:
                completed = tl.atomic_load(counter, sem="acquire", scope="gpu")
                if completed == (epoch_bits | 64):
                    consumers = tl.arange(0, 64)
                    tl.atomic_xchg(
                        Flags + (block * 64 + consumers) * 32,
                        tl.full((64,), tag, tl.int32),
                        sem="release",
                        scope="gpu",
                    )
    tl.store(Status + pid, ok)
    if allowed:
        guard.finish(Failure, ok, pid, 5)


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
    Epochs,
    Flags,
    Record,
    Polls,
    GRID: tl.constexpr,
    Seed=None,
):
    pid = tl.program_id(0)
    tag = tl.load(Epochs) + 1
    tasks = tl.load(NumPost) // 16 * 64
    for task in range(pid, tasks, GRID):
        block = task // 64
        expert = tl.load(Experts + block)
        ok, retries, observed = wait_flag(Flags + task * 32, tag)
        tl.store(Polls + task, retries)
        if ok:
            guard.delay(Seed, task, 1)
            # Cached A is legal after the explicit acquire + CTA rendezvous.
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
                CROSS_LAUNCH_A=True,
            )
        else:
            guard.finish(Failure, False, pid, 5)
            tl.store(Status + pid, 0)
            owner = tl.atomic_cas(Record, 0, 1, sem="relaxed", scope="gpu")
            if owner == 0:
                tl.store(Record + 1, pid)
                tl.store(Record + 2, task)
                tl.store(Record + 3, tag)
                tl.store(Record + 4, observed)
                tl.store(Record + 5, retries)


class Workspace(msgspec.Struct):
    epoch: object
    counters: object
    flags: object
    record: object
    polls: object
    stream: object


def make_workspace(epoch):
    device = epoch.device
    return Workspace(
        epoch,
        torch.zeros((64 * 16,), device=device, dtype=torch.int64),
        torch.zeros((4096 * 32,), device=device, dtype=torch.int32),
        torch.zeros((8,), device=device, dtype=torch.int32),
        torch.zeros((4096,), device=device, dtype=torch.int32),
        torch.cuda.Stream(device=device),
    )

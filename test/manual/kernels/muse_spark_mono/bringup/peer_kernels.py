"""Unexecuted bring-up kernels; production scoped helpers, no cache-wide fences.

The generic 2/4-rank exchange uses the same per-consumer 128B flag lines and
payload-before-flag contract. The full layer uses production reduce_row itself.
"""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_memory as mem,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_standin import (
    clock,
)


@triton.jit
def wait(Flag, tag):
    start = clock()
    seen = mem.load_flag(Flag)
    polls = 1
    elapsed = start - start
    while (seen != tag) & (polls < 65536) & (elapsed < 100_000_000):
        tl.inline_asm_elementwise(
            "s_sleep 1\nv_mov_b32 $0, 0", "=v", [], tl.int32, False, 1
        )
        seen = mem.load_flag(Flag)
        polls += 1
        if polls % 64 == 0:
            elapsed = clock() - start
    return seen == tag, seen, polls


@triton.jit
def ping(Scratch, Peer, Status, RANK: tl.constexpr):
    lane = tlx.require_layout(tl.arange(0, 128), ROW1)
    target = tl.load(Peer + (1 - RANK)).to(tl.pointer_type(tl.int32))
    if RANK == 0:
        mem.store_words(target, tl.full((128,), 73, tl.int32), lane == 0)
        mem.drain_stores()
        ok, seen, polls = wait(Scratch, 91)
    else:
        ok, seen, polls = wait(Scratch, 73)
        if ok:
            mem.store_words(target, tl.full((128,), 91, tl.int32), lane == 0)
            mem.drain_stores()
    tl.store(Status, ok)
    tl.store(Status + 1, seen)
    tl.store(Status + 2, polls)


@triton.jit
def oneshot(
    X,
    Scratch,
    Peer,
    Y,
    Status,
    WIDTH: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    TAG: tl.constexpr,
    PAY: tl.constexpr,
    FL: tl.constexpr,
):
    row = tl.program_id(0)
    words = mem.load_contiguous_words(
        X.to(tl.pointer_type(tl.int32)) + row * (WIDTH // 2), WIDTH // 2
    )
    for dst in tl.static_range(WORLD):
        peer = tl.load(Peer + dst).to(tl.pointer_type(tl.int32))
        offset = (
            PAY[0]
            + dst * PAY[1]
            + (TAG % 2) * PAY[2]
            + RANK * 4 * WIDTH
            + row * (WIDTH // 2)
        )
        mem.store_contiguous_words(peer + offset, words, WIDTH // 2)
    mem.drain_stores()
    lane = tlx.require_layout(tl.arange(0, 128), ROW1)
    for dst in tl.static_range(WORLD):
        peer = tl.load(Peer + dst).to(tl.pointer_type(tl.int32))
        offset = FL[0] + dst * FL[1] + (TAG % 2) * FL[2] + (RANK * 8 + row) * 32
        mem.store_words(peer + offset, tl.full((128,), TAG, tl.int32), lane == 0)
    mem.drain_stores()
    ok = tl.full((), True, tl.int1)
    max_polls = tl.full((), 0, tl.int32)
    for src in tl.static_range(WORLD):
        flag = Scratch + FL[0] + RANK * FL[1] + (TAG % 2) * FL[2] + (src * 8 + row) * 32
        ready, seen, polls = wait(flag, TAG)
        tl.store(Status + row * (WORLD + 1) + src + 1, seen)
        max_polls = tl.maximum(max_polls, polls)
        ok &= ready
    if ok:
        total = tl.full((WIDTH,), 0, tl.float32)
        for src in tl.static_range(WORLD):
            offset = (
                PAY[0]
                + RANK * PAY[1]
                + (TAG % 2) * PAY[2]
                + src * 4 * WIDTH
                + row * (WIDTH // 2)
            )
            packed = mem.load_contiguous_words(Scratch + offset, WIDTH // 2)
            lo = (packed & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True)
            hi = (packed >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
            total += tl.join(lo, hi).reshape((WIDTH,)).to(tl.float32)
        tl.store(Y + row * WIDTH + tl.arange(0, WIDTH), total)
    tl.store(Status + row * (WORLD + 1), tl.where(ok, max_polls, -max_polls))

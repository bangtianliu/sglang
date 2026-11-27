"""Experimental idle CTA roles warming only the next projection's prologue.

Eight CTAs run the unchanged row boundary. Other CTAs read immutable weights;
no flag replaces a data handoff. Keep launch completion before the consumer.
"""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW8,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_attn_tail import (
    s3_boundary as _s3_device,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_tail import (
    finish as _s6_device,
)


@triton.jit
def head(
    W,
    Checks,
    N: tl.constexpr,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    GRID: tl.constexpr,
    HEAD_SLEEP: tl.constexpr,
):
    pid = tl.program_id(0) - 8
    for _ in range(HEAD_SLEEP):
        tl.inline_asm_elementwise(
            "s_sleep 1\nv_mov_b32 $0,0", "=v", [], dtype=tl.int32, is_pure=False, pack=1
        )
    i = tlx.require_layout(tl.arange(0, 4096), ROW8)
    total = tl.full((4096,), 0, tl.uint32)
    for block in range(pid, tl.cdiv(N * WIDTH, 4096), GRID):
        at = block * 4096 + i
        value = tl.load(W + (at // WIDTH) * K + at % WIDTH, at < N * WIDTH, 0)
        total += value.to(tl.uint16, bitcast=True).to(tl.uint32)
    tl.store(Checks + pid, tl.sum(total, 0))


@triton.jit
def s3(
    Partial,
    Stream,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    Status,
    Epochs,
    Scratch,
    Peers,
    Reduced,
    Failure,
    GRID: tl.constexpr,
    RANK: tl.constexpr,
    PAYLOAD: tl.constexpr,
    FLAGS: tl.constexpr,
    Seed=None,
    PIN_LAYOUT: tl.constexpr = True,
    CARRIER_RULE: tl.constexpr = False,
    STORE_REDUCED: tl.constexpr = True,
    WarmW=None,
    Checks=None,
    WARM_N: tl.constexpr = 4608,
    WARM_K: tl.constexpr = 8192,
    WIDTH: tl.constexpr = 1536,
    WARM_GRID: tl.constexpr = 184,
    HEAD_SLEEP: tl.constexpr = 32,
):
    if tl.program_id(0) < 8:
        _s3_device(
            Partial,
            Stream,
            Pre,
            Alpha,
            Beta,
            Out,
            Updated,
            Status,
            Epochs,
            Scratch,
            Peers,
            Reduced,
            Failure,
            GRID,
            RANK,
            PAYLOAD,
            FLAGS,
            Seed,
            PIN_LAYOUT,
            CARRIER_RULE,
            STORE_REDUCED,
        )
    else:
        head(WarmW, Checks, WARM_N, WARM_K, WIDTH, WARM_GRID, HEAD_SLEEP)


@triton.jit
def s6(
    P,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    Status,
    Failure,
    PRE_BF16: tl.constexpr,
    PIN_LAYOUT: tl.constexpr = True,
    PACKED_INPUT: tl.constexpr = False,
    WarmW=None,
    Checks=None,
    WARM_N: tl.constexpr = 4608,
    WARM_K: tl.constexpr = 8192,
    WIDTH: tl.constexpr = 1536,
    WARM_GRID: tl.constexpr = 184,
    HEAD_SLEEP: tl.constexpr = 32,
):
    if tl.program_id(0) < 8:
        _s6_device(
            P,
            Stream,
            Post,
            Pre,
            Alpha,
            Beta,
            Out,
            Updated,
            Status,
            Failure,
            PRE_BF16,
            PIN_LAYOUT,
            PACKED_INPUT,
        )
    else:
        head(WarmW, Checks, WARM_N, WARM_K, WIDTH, WARM_GRID, HEAD_SLEEP)

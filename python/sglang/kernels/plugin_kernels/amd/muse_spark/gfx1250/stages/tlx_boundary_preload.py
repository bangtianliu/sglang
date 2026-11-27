"""T11 unselected early-gamma boundary prototype; not selected by the chain."""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_memory as mem,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
)


@triton.jit
def row_load(Base, index, PIN_LAYOUT: tl.constexpr):
    if PIN_LAYOUT:
        index = tlx.require_layout(index, ROW1, pin=True)
        value = tlx.buffer_load(Base, index, contiguity=1)
        return value
    else:
        return tl.load(Base + index)


@triton.jit
def norm_residual(
    row,
    Branch,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    Out,
    NewStream,
    POST_SCALE: tl.constexpr,
    PRE_BF16: tl.constexpr,
    Value=None,
    PIN_LAYOUT: tl.constexpr = True,
    CARRIER_RULE: tl.constexpr = False,
    BRANCH_BF16: tl.constexpr = False,
    OUTPUT_AGENT: tl.constexpr = False,
):
    i = tlx.require_layout(tl.arange(0, 8192), ROW1, pin=PIN_LAYOUT)
    if Value is not None:
        branch = tlx.require_layout(Value, ROW1)
    elif BRANCH_BF16:
        branch = row_load(Branch, row * 8192 + i, PIN_LAYOUT).to(tl.float32)
    else:
        branch = mem.load_contiguous_words(Branch + row * 8192, 8192).to(
            tl.float32, bitcast=True
        )
    if Pre is not None:
        pre_weight = row_load(Pre, i, PIN_LAYOUT)
    branch = branch.to(tl.bfloat16).to(tl.float32)
    if POST_SCALE:
        branch = (
            (branch * row_load(Post, i, PIN_LAYOUT).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
    if CARRIER_RULE:
        branch = tlx.require_layout(branch, ROW1)
    branch = tlx.require_layout(branch, ROW1, pin=PIN_LAYOUT)
    branch *= tl.rsqrt(tl.sum(branch * branch, 0) / 8192 + 1e-8)
    branch = branch.to(tl.bfloat16).to(tl.float32)
    stream = row_load(Stream, row * 8192 + i, PIN_LAYOUT)
    updated = (
        row_load(Alpha, i, PIN_LAYOUT) * stream + row_load(Beta, i, PIN_LAYOUT) * branch
    )
    tl.store(NewStream + row * 8192 + i, updated)
    pre = updated.to(tl.bfloat16).to(tl.float32) if PRE_BF16 else updated
    if CARRIER_RULE:
        pre = tlx.require_layout(pre, ROW1)
    pre = tlx.require_layout(pre, ROW1, pin=PIN_LAYOUT)
    pre *= tl.rsqrt(tl.sum(pre * pre, 0) / 8192 + 1e-5)
    if Pre is not None:
        pre *= pre_weight.to(tl.float32)
    tl.store(
        Out + row * 8192 + i,
        pre.to(tl.bfloat16),
        cache_modifier=".cg" if OUTPUT_AGENT else "",
    )

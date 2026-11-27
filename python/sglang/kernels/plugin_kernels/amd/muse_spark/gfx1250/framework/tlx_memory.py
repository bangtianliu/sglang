"""TLX versions of the audited scoped/b128 memory contract, four-wave CTAs.

Share only compile-time ISA-string generators with the Gluon implementation.
No Gluon JIT function is called from a TLX kernel.
"""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_asm import (
    _load_asm,
    _store_asm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
    ROW2,
    ROW4,
    ROW8,
)


@triton.jit
def store_words(ptr, values, mask):
    """Protect store operands with XCNT; caller drains stores before publication."""
    return tl.inline_asm_elementwise(
        "v_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\nglobal_store_b32 $1, $2, off scope:SCOPE_DEV\ns_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,v,v,v,~{scc}",
        args=[ptr, values, mask.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def load_words(ptr):
    """Complete 32-bit agent-visible loads before returning, bypassing lower caches."""
    return tl.inline_asm_elementwise(
        "global_load_b32 $0, $1, off scope:SCOPE_DEV\ns_wait_loadcnt 0\ns_wait_xcnt 0x0",
        constraints="=v,v",
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def load_flag(flag):
    """Load one scoped word in lane zero and broadcast its value through CTA reduction."""
    lanes = tlx.require_layout(tl.arange(0, 4 * 32), ROW1)
    saved_exec, value = tl.inline_asm_elementwise(
        "v_mov_b32 $1, 0\nv_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\nglobal_load_b32 $1, $2, off scope:SCOPE_DEV\ns_wait_loadcnt 0\ns_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,=&v,v,v,~{scc}",
        args=[
            tlx.release_layout(flag + tl.full(lanes.shape, 0, lanes.dtype)),
            tlx.release_layout((lanes == 0).to(tl.int32)),
        ],
        dtype=(tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    return tl.sum(value, 0)


@triton.jit
def drain_stores():
    """Wait for this wave's scoped stores, then join every wave in the CTA."""
    tl.inline_asm_elementwise(
        "s_wait_storecnt 0\nv_mov_b32 $0, 0",
        "=v",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tlx.workgroup_barrier()


@triton.jit
def load_words_masked(ptr, mask):
    """Complete masked agent-visible word loads, returning zero for inactive lanes."""
    saved_exec, value = tl.inline_asm_elementwise(
        "v_mov_b32 $1, 0\nv_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\nglobal_load_b32 $1, $2, off scope:SCOPE_DEV\ns_wait_loadcnt 0\ns_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,=&v,v,v,~{scc}",
        args=[tlx.release_layout(ptr), tlx.release_layout(mask.to(tl.int32))],
        dtype=(tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    return tlx.require_layout(value, ROW1)


@triton.jit
def split4(values, Q: tl.constexpr):
    matrix = tl.permute(tl.reshape(values, (Q, 2, 2)), (0, 2, 1))
    left, right = tl.split(matrix)
    a, b = tl.split(left)
    c, d = tl.split(right)
    layout: tl.constexpr = ROW1
    return (
        tlx.require_layout(a, layout),
        tlx.require_layout(b, layout),
        tlx.require_layout(c, layout),
        tlx.require_layout(d, layout),
    )


@triton.jit
def join4(a, b, c, d, Q: tl.constexpr):
    matrix = tl.permute(tl.join(tl.join(a, b), tl.join(c, d)), (0, 2, 1))
    return tlx.require_layout(tl.reshape(matrix, (4 * Q,)), ROW4)


@triton.jit
def join8(a, b, c, d, e, f, g, h, Q: tl.constexpr):
    matrix = tl.join(
        tl.join(tl.join(a, b), tl.join(c, d)), tl.join(tl.join(e, f), tl.join(g, h))
    )
    return tlx.require_layout(
        tl.reshape(tl.permute(matrix, (0, 3, 2, 1)), (8 * Q,)), ROW8
    )


@triton.jit
def load_contiguous_words(Base, N: tl.constexpr, RETURN_LAYOUT: tl.constexpr = ROW1):
    """Full aligned row only: quad pointers avoid pointer-transpose legalization."""
    q = tlx.require_layout(tl.arange(0, N // 4), ROW1)
    ptr = Base + q * 4
    mask = tl.full((N // 4,), 1, tl.int32)
    PACK: tl.constexpr = min(8, max(1, (N // 4) // 128))
    asm: tl.constexpr = _load_asm(1, PACK)
    a, b, c, d, _ = tl.inline_asm_elementwise(
        asm[0],
        asm[1],
        [tlx.release_layout(ptr), mask],
        dtype=(tl.int32,) * 5,
        is_pure=False,
        pack=PACK,
    )
    a = tlx.require_layout(a, ROW1)
    b = tlx.require_layout(b, ROW1)
    c = tlx.require_layout(c, ROW1)
    d = tlx.require_layout(d, ROW1)
    return tlx.require_layout(join4(a, b, c, d, N // 4), RETURN_LAYOUT)


@triton.jit
def store_contiguous_words(Base, Value, N: tl.constexpr):
    q = tlx.require_layout(tl.arange(0, N // 4), ROW1)
    ptr = Base + q * 4
    a, b, c, d = split4(tlx.require_layout(Value.reshape((N,)), ROW4), N // 4)
    mask = tl.full((N // 4,), 1, tl.int32)
    PACK: tl.constexpr = min(8, max(1, (N // 4) // 128))
    asm: tl.constexpr = _store_asm(1, PACK, True)
    tl.inline_asm_elementwise(
        asm[0],
        asm[1],
        [a, b, c, d, ptr, mask],
        dtype=tl.int32,
        is_pure=False,
        pack=PACK,
    )


@triton.jit
def store_bf16_vector(Base, Value, N: tl.constexpr):
    v = tlx.require_layout(Value.to(tl.bfloat16), ROW2)
    lo, hi = tl.split(v.reshape((N // 2, 2)))
    bits = lo.to(tl.uint16, bitcast=True).to(tl.uint32) | (
        hi.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
    )
    store_contiguous_words(
        Base.to(tl.pointer_type(tl.int32)), bits.to(tl.int32), N // 2
    )

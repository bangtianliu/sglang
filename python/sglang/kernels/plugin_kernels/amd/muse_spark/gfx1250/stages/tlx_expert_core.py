"""TLX four-wave A16W4 expert core, native FP4 hardware upcast, two LDS banks.

Only compute uses explicit layouts; output stores stay compiler-managed.
The builtin binds stock Triton's AMD scaled-upcast op through uTLX's builder,
without invoking a Gluon device operation or quantizing activations.
"""

import triton
import triton.language as tl
import utlx_plugin as tlx
from triton.language.core import _normalize_cache_policy, builtin
from utlx_plugin.layout_ops import _carrier

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    BlockedEncoding,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_attention import (
    Slice,
    fixed,
)


class ExpertMma(tlx.layout_encoding):
    def __init__(self, operand=-1, width=8):
        self.operand = operand
        self.width = width

    def __repr__(self):
        return f"ExpertMma4_{self.operand}_{self.width}"

    def __hash__(self):
        return hash((ExpertMma, self.operand, self.width))

    def __eq__(self, other):
        return isinstance(other, ExpertMma) and (self.operand, self.width) == (
            other.operand,
            other.width,
        )

    def to_ir(self, builder, *_):
        parent = builder.get_amd_wmma_layout(
            3, True, [[0, 1], [0, 2]], [], [], [16, 16, 32], 2
        )
        return (
            parent
            if self.operand < 0
            else builder.get_dot_operand_layout(self.operand, parent, self.width)
        )


class ExpandedScale(tlx.layout_encoding):
    def __repr__(self):
        return "ExpandedExpertScale"

    def __hash__(self):
        return hash(ExpandedScale)

    def __eq__(self, other):
        return isinstance(other, ExpandedScale)

    def to_ir(self, builder, *_):
        return builder.get_distributed_linear_layout(
            [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 8, 0]],
            [[0, 0, 16], [0, 0, 32]],
            [],
            [4, 32, 64],
        )


MMA = tl.constexpr(ExpertMma())
A_LAYOUT = tl.constexpr(ExpertMma(0))
B_PACKED = tl.constexpr(ExpertMma(1, 4))
B_EXPANDED = tl.constexpr(ExpertMma(1, 8))
SCALE_LAYOUT = tl.constexpr(Slice(1, ExpandedScale()))
AL = tl.constexpr(BlockedEncoding([1, 8], [2, 16], [4, 1], [1, 0]))
BL = tl.constexpr(BlockedEncoding([16, 1], [4, 8], [1, 4], [0, 1]))
SL = tl.constexpr(BlockedEncoding([4, 1], [1, 32], [1, 4], [0, 1]))
WORD_LOAD = tl.constexpr(BlockedEncoding([1, 1], [1, 32], [1, 4], [1, 0]))


@builtin
def fp4_upcast(packed, expanded, _semantic=None):
    assert packed.dtype == tl.uint8 and expanded.dtype == tl.uint8
    handle = _semantic.builder.create_scaled_upcast_fp4(
        packed.handle, expanded.handle, tl.bfloat16.to_ir(_semantic.builder), 0
    )
    return _carrier(handle, tl.bfloat16)


@triton.jit
def weight(sb, ss, bank):
    packed = tlx.local_load(tlx.local_view(sb, bank), layout=B_PACKED)
    scales = tlx.local_load(tlx.local_view(ss, bank), layout=SCALE_LAYOUT)
    expanded = tl.broadcast_to(scales[:, None, :], (4, 32, 64)).reshape((128, 64))
    expanded = tlx.require_layout(expanded, B_EXPANDED)
    return fp4_upcast(packed, expanded)


@triton.jit
def copy_scale(ptr, dst, SCALE_ASYNC: tl.constexpr = False, Word=None):
    if SCALE_ASYNC:
        # Four contiguous E8M0 bytes per neuron, naturally aligned at every
        # K128 step. Reinterpret the same LDS bytes; no weight repacking.
        word_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], alignment=16
        )
        word_dst = tlx.local_reinterpret(dst, tl.uint32, [1, 64], layout=word_layout)
        words = Word
        words = tlx.require_layout(words, WORD_LOAD)
        words = tl.multiple_of(words, (1, 4))
        tlx.async_load(words, word_dst)
    else:
        ptr = tlx.require_layout(ptr, SL)
        ptr = tl.max_contiguous(tl.multiple_of(ptr, (4, 1)), (4, 1))
        tlx.local_store(dst, tl.load(ptr))


@builtin
def async_agent(src, dest, mask, other, _semantic=None):
    # uTLX 6c03c3f validates async_load cache hints but drops them. Bind the
    # stock typed async-copy builder directly, preserving the .cg policy.
    _, src, mask, other, _ = _semantic._prepare_legacy_load(
        src, mask, other, None, None
    )
    policy = _normalize_cache_policy(None, ".cg", None)._to_ir(_semantic.builder)
    _semantic.builder.create_async_copy_global_to_local(
        dest.handle, src.handle, mask.handle, other.handle, policy, False
    )


@triton.jit
def prefetch(
    X,
    W,
    Scale,
    ap,
    bp,
    sp,
    spw,
    step,
    bank,
    sa,
    sb,
    ss,
    mask,
    FIRST: tl.constexpr,
    WEIGHT_CACHE: tl.constexpr,
    SCALE_ASYNC: tl.constexpr,
    CROSS_LAUNCH_A: tl.constexpr,
):
    if FIRST:
        tlx.async_load(
            X + ap + step * 128,
            tlx.local_view(sa, bank),
            mask,
            tlx.require_layout(tl.full((16, 128), 0.0, tl.bfloat16), AL),
        )
    elif CROSS_LAUNCH_A:
        # An ordered W13 dispatch produced X; match native's default-cache DMA.
        # Direct/in-kernel core users retain device scope via the default below.
        tlx.async_load(
            X + ap + step * 128,
            tlx.local_view(sa, bank),
            mask,
            tlx.require_layout(tl.full((16, 128), 0.0, tl.bfloat16), AL),
        )
    else:
        async_agent(
            X + ap + step * 128,
            tlx.local_view(sa, bank),
            mask,
            tlx.require_layout(tl.full((16, 128), 0.0, tl.bfloat16), AL),
        )
    if WEIGHT_CACHE == "":
        tlx.async_load(W + bp + step * 64, tlx.local_view(sb, bank))
    else:
        tlx.local_store(
            tlx.local_view(sb, bank),
            tl.load(W + bp + step * 64, cache_modifier=WEIGHT_CACHE),
        )
    copy_scale(
        Scale + sp + step * 4,
        tlx.local_view(ss, bank),
        SCALE_ASYNC,
        Scale.to(tl.pointer_type(tl.uint32)) + spw + step,
    )


@triton.jit
def expert_tile(
    X,
    Sorted,
    W,
    Scale,
    Out,
    block,
    expert,
    tile,
    FIRST: tl.constexpr,
    WEIGHT_CACHE: tl.constexpr = "",
    SCALE_ASYNC: tl.constexpr = False,
    CROSS_LAUNCH_A: tl.constexpr = False,
    OUTPUT_AGENT: tl.constexpr = False,
):
    row = tl.arange(0, 16)
    ak = tl.arange(0, 128)
    slots = tl.load(Sorted + block * 16 + row)
    input_rows = slots // 8 if FIRST else slots
    ap = tlx.require_layout(input_rows[:, None] * 4096 + ak[None, :], AL)
    mask = tlx.require_layout(tl.broadcast_to(slots[:, None] < 64, (16, 128)), AL)
    bk = tl.arange(0, 64)
    bn = tl.arange(0, 64) + tile * 64
    sk = tl.arange(0, 4)
    sn = tl.arange(0, 64) + tile * 64
    N: tl.constexpr = 8192 if FIRST else 4096
    bp = tlx.require_layout(
        expert.to(tl.int64) * N * 2048 + bn[None, :] * 2048 + bk[:, None], BL
    )
    sp = tlx.require_layout(
        expert.to(tl.int64) * N * 128 + sn[None, :] * 128 + sk[:, None], SL
    )
    spw = expert.to(tl.int64) * N * 32 + sn[None, :] * 32
    ash: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(128, 8)], [16, 128], [1, 0]
    )
    bsh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(256, 16)], [64, 64], [0, 1]
    )
    ssh: tl.constexpr = tlx.shared_linear_layout_encoding(
        [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], alignment=16
    )
    sa = tlx.local_alloc((16, 128), tl.bfloat16, 2, layout=ash)
    sb = tlx.local_alloc((64, 64), tl.uint8, 2, layout=bsh)
    ss = tlx.local_alloc((4, 64), tl.uint8, 2, layout=ssh)
    if FIRST:
        su = tlx.local_alloc((64, 64), tl.uint8, 2, layout=bsh)
        st = tlx.local_alloc((4, 64), tl.uint8, 2, layout=ssh)
    prefetch(
        X,
        W,
        Scale,
        ap,
        bp,
        sp,
        spw,
        0,
        0,
        sa,
        sb,
        ss,
        mask,
        FIRST,
        WEIGHT_CACHE,
        SCALE_ASYNC,
        CROSS_LAUNCH_A,
    )
    if FIRST:
        if WEIGHT_CACHE == "":
            tlx.async_load(W + bp + 4096 * 2048, tlx.local_view(su, 0))
        else:
            tlx.local_store(
                tlx.local_view(su, 0),
                tl.load(W + bp + 4096 * 2048, cache_modifier=WEIGHT_CACHE),
            )
        copy_scale(
            Scale + sp + 4096 * 128,
            tlx.local_view(st, 0),
            SCALE_ASYNC,
            Scale.to(tl.pointer_type(tl.uint32)) + spw + 4096 * 32,
        )
    tlx.async_load_commit_group()
    acc = fixed(tl.zeros((16, 64), tl.float32), MMA)
    if FIRST:
        up = fixed(tl.zeros((16, 64), tl.float32), MMA)
    for step in range(32):
        bank = step % 2
        tlx.async_load_wait_group(0)
        tlx.workgroup_barrier()
        if step + 1 < 32:
            prefetch(
                X,
                W,
                Scale,
                ap,
                bp,
                sp,
                spw,
                step + 1,
                1 - bank,
                sa,
                sb,
                ss,
                mask,
                FIRST,
                WEIGHT_CACHE,
                SCALE_ASYNC,
                CROSS_LAUNCH_A,
            )
            if FIRST:
                if WEIGHT_CACHE == "":
                    tlx.async_load(
                        W + bp + (step + 1) * 64 + 4096 * 2048,
                        tlx.local_view(su, 1 - bank),
                    )
                else:
                    tlx.local_store(
                        tlx.local_view(su, 1 - bank),
                        tl.load(
                            W + bp + (step + 1) * 64 + 4096 * 2048,
                            cache_modifier=WEIGHT_CACHE,
                        ),
                    )
                copy_scale(
                    Scale + sp + (step + 1) * 4 + 4096 * 128,
                    tlx.local_view(st, 1 - bank),
                    SCALE_ASYNC,
                    Scale.to(tl.pointer_type(tl.uint32)) + spw + (step + 1) + 4096 * 32,
                )
            tlx.async_load_commit_group()
        a = tlx.local_load(tlx.local_view(sa, bank), layout=A_LAYOUT)
        b = weight(sb, ss, bank)
        acc = tl.dot(a, b, acc)
        if FIRST:
            up = tl.dot(a, weight(su, st, bank), up)
        tlx.workgroup_barrier()
    tlx.async_load_wait_group(0)
    if FIRST:
        acc = acc / (1.0 + tl.exp(-acc)) * up
    rows = tl.arange(0, 16)
    col = tl.arange(0, 64) + tile * 64
    output_slots = tl.load(Sorted + block * 16 + rows)
    tl.store(
        Out + output_slots[:, None] * 4096 + col[None, :],
        acc.to(tl.bfloat16),
        output_slots[:, None] < 64,
        cache_modifier=".cg" if OUTPUT_AGENT else "",
    )

"""T11 rejected three-bank W2 prototype; not selected by the chain."""

import triton
import triton.language as tl
import utlx_plugin as tlx
from triton.language.core import _normalize_cache_policy, builtin

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    BlockedEncoding,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_attention import (
    fixed,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_core import (
    A_LAYOUT,
    AL,
    B_EXPANDED,
    B_PACKED,
    BL,
    MMA,
    SCALE_LAYOUT,
    SL,
    async_agent,
    fp4_upcast,
)

SWORD = tl.constexpr(BlockedEncoding([4, 1], [1, 32], [1, 4], [0, 1]))


@triton.jit
def copy_scale(ptr, dst, SCALE_ASYNC: tl.constexpr = False, Word=None):
    if SCALE_ASYNC:
        word_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
            [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
            alignment=16,
        )
        word_dst = tlx.local_reinterpret(dst, tl.uint32, [4, 64], layout=word_layout)
        words = tlx.require_layout(
            Word.reshape((64,))[None, :] + tl.arange(0, 4)[:, None], SWORD
        )
        words = tl.max_contiguous(tl.multiple_of(words, (4, 1)), (4, 1))
        tlx.async_load(words, word_dst)
    else:
        ptr = tlx.require_layout(ptr, SL)
        tlx.local_store(dst, tl.load(ptr))


@builtin
def async_nt(src, dest, mask, other, _semantic=None):
    _, src, mask, other, _ = _semantic._prepare_legacy_load(
        src, mask, other, None, None
    )
    policy = _normalize_cache_policy(None, ".cs", None)._to_ir(_semantic.builder)
    _semantic.builder.create_async_copy_global_to_local(
        dest.handle, src.handle, mask.handle, other.handle, policy, False
    )


@triton.jit
def weight_nt(src, dest):
    mask = tlx.require_layout(tl.full((256, 64), True, tl.int1), BL)
    zero = tlx.require_layout(tl.full((256, 64), 0, tl.uint8), BL)
    async_nt(src, dest, mask, zero)


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
            X + ap + step * 512,
            tlx.local_view(sa, bank),
            mask,
            tlx.require_layout(tl.full((16, 512), 0.0, tl.bfloat16), AL),
        )
    elif CROSS_LAUNCH_A:
        tlx.async_load(
            X + ap + step * 512,
            tlx.local_view(sa, bank),
            mask,
            tlx.require_layout(tl.full((16, 512), 0.0, tl.bfloat16), AL),
        )
    else:
        async_agent(
            X + ap + step * 512,
            tlx.local_view(sa, bank),
            mask,
            tlx.require_layout(tl.full((16, 512), 0.0, tl.bfloat16), AL),
        )
    if WEIGHT_CACHE == "":
        tlx.async_load(W + bp + step * 256, tlx.local_view(sb, bank))
    elif WEIGHT_CACHE == ".cs":
        weight_nt(W + bp + step * 256, tlx.local_view(sb, bank))
    else:
        tlx.local_store(
            tlx.local_view(sb, bank),
            tl.load(W + bp + step * 256, cache_modifier=WEIGHT_CACHE),
        )
    copy_scale(
        Scale + sp + step * 16,
        tlx.local_view(ss, bank),
        SCALE_ASYNC,
        Scale.to(tl.pointer_type(tl.uint32)) + spw + step * 4,
    )


@triton.jit
def weight_sub(sb, ss, bank, sub: tl.constexpr):
    packed = tlx.local_load(
        tlx.local_slice(tlx.local_view(sb, bank), [sub * 64, 0], [64, 64]),
        layout=B_PACKED,
    )
    scales = tlx.local_load(
        tlx.local_slice(tlx.local_view(ss, bank), [sub * 4, 0], [4, 64]),
        layout=SCALE_LAYOUT,
    )
    expanded = tl.broadcast_to(scales[:, None, :], (4, 32, 64)).reshape((128, 64))
    expanded = tlx.require_layout(expanded, B_EXPANDED)
    return fp4_upcast(packed, expanded)


@triton.jit
def expert_tile_wide(
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
    SCALE_ASYNC: tl.constexpr = True,
    CROSS_LAUNCH_A: tl.constexpr = False,
    OUTPUT_AGENT: tl.constexpr = False,
    UNUSED_DEPTH: tl.constexpr = 2,
    PREFETCH_FIRST: tl.constexpr = True,
):
    DEPTH: tl.constexpr = 2 if FIRST else 3
    row = tl.arange(0, 16)
    ak = tl.arange(0, 512)
    slots = tl.load(Sorted + block * 16 + row)
    input_rows = slots // 8 if FIRST else slots
    ap = tlx.require_layout(input_rows[:, None] * 4096 + ak[None, :], AL)
    mask = tlx.require_layout(tl.broadcast_to(slots[:, None] < 64, (16, 512)), AL)
    bk = tl.arange(0, 256)
    bn = tl.arange(0, 64) + tile * 64
    sk = tl.arange(0, 16)
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
        [(512, 8)], [16, 512], [1, 0]
    )
    bsh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(256, 16)], [256, 64], [0, 1]
    )
    ssh: tl.constexpr = tlx.shared_linear_layout_encoding(
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
        ],
        alignment=16,
    )
    sa = tlx.local_alloc((16, 512), tl.bfloat16, DEPTH, layout=ash)
    sb = tlx.local_alloc((256, 64), tl.uint8, DEPTH, layout=bsh)
    ss = tlx.local_alloc((16, 64), tl.uint8, DEPTH, layout=ssh)
    if FIRST:
        su = tlx.local_alloc((256, 64), tl.uint8, DEPTH, layout=bsh)
        st = tlx.local_alloc((16, 64), tl.uint8, DEPTH, layout=ssh)
    for init_step in tl.static_range(DEPTH - 1):
        prefetch(
            X,
            W,
            Scale,
            ap,
            bp,
            sp,
            spw,
            init_step,
            init_step,
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
                    W + bp + init_step * 256 + 4096 * 2048,
                    tlx.local_view(su, init_step),
                )
            elif WEIGHT_CACHE == ".cs":
                weight_nt(
                    W + bp + init_step * 256 + 4096 * 2048,
                    tlx.local_view(su, init_step),
                )
            else:
                tlx.local_store(
                    tlx.local_view(su, init_step),
                    tl.load(
                        W + bp + init_step * 256 + 4096 * 2048,
                        cache_modifier=WEIGHT_CACHE,
                    ),
                )
            copy_scale(
                Scale + sp + init_step * 16 + 4096 * 128,
                tlx.local_view(st, init_step),
                SCALE_ASYNC,
                Scale.to(tl.pointer_type(tl.uint32)) + spw + init_step * 4 + 4096 * 32,
            )
        tlx.async_load_commit_group()
    acc = fixed(tl.zeros((16, 64), tl.float32), MMA)
    if FIRST:
        up = fixed(tl.zeros((16, 64), tl.float32), MMA)
    for step in range(8):
        bank = step % DEPTH
        if not PREFETCH_FIRST:
            tlx.async_load_wait_group(0)
            tlx.workgroup_barrier()
        if "full" != "compute" and step + DEPTH - 1 < 8:
            prefetch(
                X,
                W,
                Scale,
                ap,
                bp,
                sp,
                spw,
                step + DEPTH - 1,
                (step + DEPTH - 1) % DEPTH,
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
                        W + bp + (step + DEPTH - 1) * 256 + 4096 * 2048,
                        tlx.local_view(su, (step + DEPTH - 1) % DEPTH),
                    )
                elif WEIGHT_CACHE == ".cs":
                    weight_nt(
                        W + bp + (step + DEPTH - 1) * 256 + 4096 * 2048,
                        tlx.local_view(su, (step + DEPTH - 1) % DEPTH),
                    )
                else:
                    tlx.local_store(
                        tlx.local_view(su, (step + DEPTH - 1) % DEPTH),
                        tl.load(
                            W + bp + (step + DEPTH - 1) * 256 + 4096 * 2048,
                            cache_modifier=WEIGHT_CACHE,
                        ),
                    )
                copy_scale(
                    Scale + sp + (step + DEPTH - 1) * 16 + 4096 * 128,
                    tlx.local_view(st, (step + DEPTH - 1) % DEPTH),
                    SCALE_ASYNC,
                    Scale.to(tl.pointer_type(tl.uint32))
                    + spw
                    + (step + DEPTH - 1) * 4
                    + 4096 * 32,
                )
            tlx.async_load_commit_group()
        if PREFETCH_FIRST:
            if step + DEPTH - 1 < 8:
                tlx.async_load_wait_group(DEPTH - 1)
            else:
                tlx.async_load_wait_group(0)
            tlx.workgroup_barrier()
        for sub in tl.static_range(4):
            av = tlx.local_slice(tlx.local_view(sa, bank), [0, sub * 128], [16, 128])
            a = tlx.local_load(av, layout=A_LAYOUT)
            b = weight_sub(sb, ss, bank, sub)
            acc = tl.dot(a, b, acc)
            if FIRST:
                up = tl.dot(a, weight_sub(su, st, bank, sub), up)
        tlx.workgroup_barrier()
    tlx.async_load_wait_group(0)
    if FIRST:
        acc = acc / (1.0 + tl.exp(-acc)) * up
    rows = tl.arange(0, 16)
    col = tl.arange(0, 64) + tile * 64
    output_slots = tl.load(Sorted + block * 16 + rows)
    output_mask = output_slots < 64
    tl.store(
        Out + output_slots[:, None] * 4096 + col[None, :],
        acc.to(tl.bfloat16),
        output_mask[:, None],
        cache_modifier=".cg" if OUTPUT_AGENT else "",
    )

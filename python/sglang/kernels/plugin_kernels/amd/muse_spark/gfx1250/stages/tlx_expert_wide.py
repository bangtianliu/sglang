"""TLX canonical K512 expert loads and four ordered K128 WMMA subtiles."""

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
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_standin import (
    clock as tlx_clock,
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
    MX: tl.constexpr = False,
    DO_A: tl.constexpr = True,
    DO_W: tl.constexpr = True,
):
    if not DO_A:
        pass
    elif MX:
        tlx.async_load(
            X + ap + step * 512,
            tlx.local_view(sa, bank),
            mask,
            tlx.require_layout(tl.full((16, 512), 0, tl.uint8), AL),
        )
    elif FIRST:
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
    if not DO_W:
        pass
    elif WEIGHT_CACHE == "":
        tlx.async_load(W + bp + step * 256, tlx.local_view(sb, bank))
    elif WEIGHT_CACHE == ".cs":
        weight_nt(W + bp + step * 256, tlx.local_view(sb, bank))
    else:
        tlx.local_store(
            tlx.local_view(sb, bank),
            tl.load(W + bp + step * 256, cache_modifier=WEIGHT_CACHE),
        )
    if DO_W:
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
def quant_mx(x, R: tl.constexpr, C: tl.constexpr):
    """MXFP8: x [R, C] fp32 -> e4m3 values [R, C] and e8m0 scales [R, C // 32]
    (one power-of-two scale per 32 along C, chosen so |x| / scale <= 448)."""
    xb = tl.reshape(x, (R, C // 32, 32))
    amax = tl.max(tl.abs(xb), 2)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    se = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    se = tl.minimum(se, 253)
    inv = ((254 - se) << 23).to(tl.float32, bitcast=True)
    y = xb * inv[:, :, None]
    q = tl.minimum(tl.maximum(y, -448.0), 448.0).to(tl.float8e4nv)
    return tl.reshape(q, (R, C)), se.to(tl.uint8)


@triton.jit
def quant_mx_back(x, R: tl.constexpr, C: tl.constexpr):
    """quant_mx that also returns the value the fp8 term stands for (for the residual term)"""
    xb = tl.reshape(x, (R, C // 32, 32))
    amax = tl.max(tl.abs(xb), 2)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    se = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    se = tl.minimum(se, 253)
    inv = ((254 - se) << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(xb * inv[:, :, None], -448.0), 448.0).to(tl.float8e4nv)
    back = q.to(tl.float32) * (se << 23).to(tl.float32, bitcast=True)[:, :, None]
    return tl.reshape(q, (R, C)), se.to(tl.uint8), tl.reshape(back, (R, C))


@triton.jit
def mx_step2(sa, asc, sa2, asc2, sb, ss, bank, acc):
    """hi and lo activation terms against the same MXFP4 weight tile"""
    a = tlx.local_load(tlx.local_view(sa, bank), layout=AL).to(tl.float8e4nv, bitcast=True)
    a2 = tlx.local_load(tlx.local_view(sa2, bank), layout=AL).to(tl.float8e4nv, bitcast=True)
    b = tlx.local_load(tlx.local_view(sb, bank), layout=BL)
    bs = tl.trans(tlx.local_load(tlx.local_view(ss, bank), layout=SL))
    acc = tl.dot_scaled(a, asc, "e4m3", b, bs, "e2m1", acc)
    return tl.dot_scaled(a2, asc2, "e4m3", b, bs, "e2m1", acc)


@triton.jit
def mx_step(sa, asc, sb, ss, bank, acc):
    aq = tlx.local_load(tlx.local_view(sa, bank), layout=AL).to(tl.float8e4nv, bitcast=True)
    ascale = asc
    b = tlx.local_load(tlx.local_view(sb, bank), layout=BL)
    bs = tlx.local_load(tlx.local_view(ss, bank), layout=SL)
    return tl.dot_scaled(aq, ascale, "e4m3", b, tl.trans(bs), "e2m1", acc)


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
    DEPTH: tl.constexpr = 2,
    PREFETCH_FIRST: tl.constexpr = True,
    SLOTS=None,
    MX: tl.constexpr = False,
    XS=None,
    QOUT: tl.constexpr = False,
    OutS=None,
    X2=None,
    XS2=None,
    Publish=None,
    Wait=None,
    WAIT_BOUND: tl.constexpr = 200000,
    POLL_SLEEP: tl.constexpr = 0,
    PUBLISH64: tl.constexpr = False,
    CNT_INDEX=None,
    NEED=None,
    FOLD: tl.constexpr = False,
):
    """Publish (int32 per-block counters): W13 output goes out by relaxed gpu-scope
    atomic_xchg (bf16 pairs as int32), and once every wave's atomics have returned,
    one atomic add on Publish[block * 32] (MK.39/40 handoff).
    Wait (W2 side): the first DEPTH-1 weight banks are issued, then Wait[block * 32] >= 64 is
    polled (bounded), then the activations are read (first touch). Returns whether it was seen."""
    row = tl.arange(0, 16)
    ak = tl.arange(0, 512)
    if FOLD:
        # MK.62: a block has at most 8 valid rows (8 tokens, an expert at most once per token),
        # packed first. Tile rows 0-7 take the hi fp8 term of those rows and rows 8-15 the lo
        # term (X / XS hold [hi rows 0-7; lo rows 8-15]), so one scaled WMMA covers both terms;
        # the epilogue adds row r and row r + 8. Exact for bf16 inputs, like the two-term split.
        # W2 (MK.67): the same fold over the slot rows, X / XS = [hi slots 0-63; lo slots 64-127]
        # as written by W13's QOUT epilogue below.
        tl.static_assert(MX and SLOTS is None and X2 is None and (FIRST or not QOUT))
        tl.static_assert(Publish is None)
        slots = tl.load(Sorted + block * 16 + row % 8)
        if FIRST:
            input_rows = slots // 8 + 8 * (row // 8)
        else:
            input_rows = slots + 64 * (row // 8)
    elif SLOTS is None:
        slots = tl.load(Sorted + block * 16 + row)
        input_rows = slots // 8 if FIRST else slots
    else:
        slots = SLOTS
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
    sa = tlx.local_alloc((16, 512), tl.uint8 if MX else tl.bfloat16, DEPTH, layout=ash)
    TWO: tl.constexpr = X2 is not None
    if TWO:
        sa2 = tlx.local_alloc((16, 512), tl.uint8, DEPTH, layout=ash)
    sb = tlx.local_alloc((256, 64), tl.uint8, DEPTH, layout=bsh)
    ss = tlx.local_alloc((16, 64), tl.uint8, DEPTH, layout=ssh)
    if FIRST:
        su = tlx.local_alloc((256, 64), tl.uint8, DEPTH, layout=bsh)
        st = tlx.local_alloc((16, 64), tl.uint8, DEPTH, layout=ssh)
    WAITS: tl.constexpr = Wait is not None
    seen = True
    if WAITS:
        for init_step in tl.static_range(DEPTH - 1):
            prefetch(X, W, Scale, ap, bp, sp, spw, init_step, init_step, sa, sb, ss, mask,
                     FIRST, WEIGHT_CACHE, SCALE_ASYNC, CROSS_LAUNCH_A, MX, DO_A=False)
        t0 = tlx_clock()
        widx = block if CNT_INDEX is None else CNT_INDEX
        need = 64 if NEED is None else NEED
        c = tl.atomic_add(Wait + widx * 32, 0, sem="relaxed", scope="gpu")
        while (c < need) & ((tlx_clock() - t0) < WAIT_BOUND):
            if POLL_SLEEP > 0:
                tl.inline_asm_elementwise(f"s_sleep {POLL_SLEEP}\nv_mov_b32 $0, 0", "=v", [],
                                          dtype=tl.int32, is_pure=False, pack=1)
            c = tl.atomic_add(Wait + widx * 32, 0, sem="relaxed", scope="gpu")
        seen = c >= need
        X = X + tl.minimum(c - need, 0) * 4096  # orders the activation reads after the poll
        for init_step in tl.static_range(DEPTH - 1):
            prefetch(X, W, Scale, ap, bp, sp, spw, init_step, init_step, sa, sb, ss, mask,
                     FIRST, WEIGHT_CACHE, SCALE_ASYNC, CROSS_LAUNCH_A, MX, DO_W=False)
        tlx.async_load_commit_group()
    for init_step in tl.static_range((DEPTH - 1) * (not WAITS)):
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
            MX,
        )
        if TWO:
            tlx.async_load(X2 + ap + init_step * 512, tlx.local_view(sa2, init_step), mask,
                           tlx.require_layout(tl.full((16, 512), 0, tl.uint8), AL))
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
    if MX:
        acc = tl.zeros((16, 64), tl.float32)
        if FIRST:
            up = tl.zeros((16, 64), tl.float32)
    else:
        acc = fixed(tl.zeros((16, 64), tl.float32), MMA)
        if FIRST:
            up = fixed(tl.zeros((16, 64), tl.float32), MMA)
    if MX:
        srow = tl.arange(0, 16)
        sj = tl.arange(0, 16)
        sptr = XS + input_rows[:, None] * 128 + sj[None, :]
        smask = slots[:, None] < 64
        if TWO:
            sptr2 = XS2 + input_rows[:, None] * 128 + sj[None, :]
    for step in range(8):
        bank = step % DEPTH
        if MX:
            asc = tl.load(sptr + step * 16, smask, other=0)
            if TWO:
                asc2 = tl.load(sptr2 + step * 16, smask, other=0)
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
                MX,
            )
            if TWO:
                tlx.async_load(X2 + ap + (step + DEPTH - 1) * 512,
                               tlx.local_view(sa2, (step + DEPTH - 1) % DEPTH), mask,
                               tlx.require_layout(tl.full((16, 512), 0, tl.uint8), AL))
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
            if WAITS and (step + 1 < DEPTH - 1):
                # the first DEPTH-1 steps live in one group (issued around the poll)
                tlx.async_load_wait_group(0)
            elif step + DEPTH - 1 < 8:
                tlx.async_load_wait_group(DEPTH - 1)
            else:
                tlx.async_load_wait_group(0)
            tlx.workgroup_barrier()
        if TWO:
            acc = mx_step2(sa, asc, sa2, asc2, sb, ss, bank, acc)
            up = mx_step2(sa, asc, sa2, asc2, su, st, bank, up)
        elif MX:
            acc = mx_step(sa, asc, sb, ss, bank, acc)
            if FIRST:
                up = mx_step(sa, asc, su, st, bank, up)
        for sub in tl.static_range(4 * (not MX)):
            av = tlx.local_slice(tlx.local_view(sa, bank), [0, sub * 128], [16, 128])
            a = tlx.local_load(av, layout=A_LAYOUT)
            b = weight_sub(sb, ss, bank, sub)
            acc = tl.dot(a, b, acc)
            if FIRST:
                up = tl.dot(a, weight_sub(su, st, bank, sub), up)
        tlx.workgroup_barrier()
    tlx.async_load_wait_group(0)
    if FOLD:
        value = tl.sum(tl.reshape(acc, (2, 8, 64)), 0)
        if FIRST:
            upf = tl.sum(tl.reshape(up, (2, 8, 64)), 0)
            value = value / (1.0 + tl.exp(-value)) * upf
        rows8 = tl.arange(0, 8)
        col8 = tl.arange(0, 64) + tile * 64
        out8 = tl.load(Sorted + block * 16 + rows8)
        ok8 = (out8 < 64)[:, None]
        if QOUT:
            # W2's activations: the bf16 value as two exact MXFP8 terms, hi at slot rows 0-63
            # and the residual at rows 64-127 (two 32-column scale blocks per row of this tile)
            v = value.to(tl.bfloat16).to(tl.float32)
            q, qs, back = quant_mx_back(v, 8, 64)
            q2, qs2, _ = quant_mx_back(v - back, 8, 64)
            qj = tl.arange(0, 2)
            tl.store(Out + out8[:, None] * 4096 + col8[None, :], q.to(tl.uint8, bitcast=True), ok8)
            tl.store(Out + (out8[:, None] + 64) * 4096 + col8[None, :],
                     q2.to(tl.uint8, bitcast=True), ok8)
            tl.store(OutS + out8[:, None] * 128 + tile * 2 + qj[None, :], qs, ok8)
            tl.store(OutS + (out8[:, None] + 64) * 128 + tile * 2 + qj[None, :], qs2, ok8)
        else:
            tl.store(Out + out8[:, None] * 4096 + col8[None, :], value.to(tl.bfloat16), ok8)
        return seen
    if FIRST:
        acc = acc / (1.0 + tl.exp(-acc)) * up
    rows = tl.arange(0, 16)
    col = tl.arange(0, 64) + tile * 64
    output_slots = tl.load(Sorted + block * 16 + rows)
    output_mask = output_slots < 64
    if QOUT:
        # W2's activations as MXFP8 (two 32-column blocks per row of this tile)
        q, qs = quant_mx(acc.to(tl.bfloat16).to(tl.float32), 16, 64)
        tl.store(
            Out + output_slots[:, None] * 4096 + col[None, :],
            q.to(tl.uint8, bitcast=True),
            output_mask[:, None],
        )
        qj = tl.arange(0, 2)
        tl.store(OutS + output_slots[:, None] * 128 + tile * 2 + qj[None, :], qs,
                 output_mask[:, None])
    elif Publish is not None:
        lo, hi = tl.split(tl.reshape(acc.to(tl.bfloat16), (16, 32, 2)))
        word = (hi.to(tl.int16, bitcast=True).to(tl.int32) << 16) | (
            lo.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF)
        if PUBLISH64:
            w0, w1 = tl.split(tl.reshape(word, (16, 16, 2)))
            word64 = (w1.to(tl.int64) << 32) | (w0.to(tl.int64) & 0xFFFFFFFF)
            wj = tl.arange(0, 16)
            old = tl.atomic_xchg(
                Out.to(tl.pointer_type(tl.int64)) + output_slots[:, None] * 1024 + tile * 16
                + wj[None, :], word64, output_mask[:, None], sem="relaxed", scope="gpu")
        else:
            wj = tl.arange(0, 32)
            old = tl.atomic_xchg(
                Out.to(tl.pointer_type(tl.int32)) + output_slots[:, None] * 2048 + tile * 32
                + wj[None, :], word, output_mask[:, None], sem="relaxed", scope="gpu")
        # the asm consumes every returned value: this wave's atomics are performed
        dep = tl.inline_asm_elementwise("v_mov_b32 $0, 0", "=v,v", [old.to(tl.int32)],
                                        dtype=tl.int32, is_pure=False, pack=1)
        dep = tl.min(tl.min(dep, 1), 0)
        tlx.workgroup_barrier()
        cidx = block if CNT_INDEX is None else CNT_INDEX
        tl.atomic_add(Publish + cidx * 32 + dep, 1, sem="relaxed", scope="gpu")
    else:
        tl.store(
            Out + output_slots[:, None] * 4096 + col[None, :],
            acc.to(tl.bfloat16),
            output_mask[:, None],
            cache_modifier=".cg" if OUTPUT_AGENT else "",
        )
    return seen

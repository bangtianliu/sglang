"""WIP Triton3.9 gfx1250 A16W4 M16/N64/K128 four-wave expert core.

Standalone W13 and W2 launches consume S4's padded16-row slot table. Two LDS
buffers stage BF16 inputs, packed FP4 weights and raw E8M0 scales. No persistent
launch policy uses this core before standalone validation and residency checks.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, cdna5, gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_bf16_tile,
    store_tile_words,
)


@g.jit
def _prefetch(
    A,
    B,
    Scale,
    ap,
    bp,
    sp,
    step,
    bank,
    sa,
    sb,
    ss,
    mask,
    COHERENT: gl.constexpr = False,
    READ_MODE: gl.constexpr = 0,
    WEIGHT_CACHE: gl.constexpr = "",
):
    if COHERENT and READ_MODE == 0:
        # In-launch W2 consumes the audited agent-scope W13 publication.
        sa.index(bank).store(load_bf16_tile(A + ap + step * 128, mask, 16, 128))
    else:
        gfx1250.async_copy.global_to_shared(
            sa.index(bank),
            A + ap + step * 128,
            mask,
            0,
            cache_modifier=".cg" if COHERENT and READ_MODE == 2 else "",
        )
    gfx1250.async_copy.global_to_shared(
        sb.index(bank), B + bp + step * 64, cache_modifier=WEIGHT_CACHE
    )
    gfx1250.async_copy.global_to_shared(ss.index(bank), Scale + sp + step * 4)


@g.jit
def _weight(sb, ss, bank, mma: gl.constexpr):
    packed = sb.index(bank).load(gl.DotOperandLayout(1, mma, 4))
    expanded_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0]],
        lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 8, 0]],
        warp_bases=[[0, 0, 16], [0, 0, 32]],
        block_bases=[],
        shape=[4, 32, 64],
    )
    scale_layout: gl.constexpr = gl.SliceLayout(1, expanded_layout)
    scales = ss.index(bank).load(scale_layout)
    expanded = scales[:, None, :].broadcast_to((4, 32, 64)).reshape((128, 64))
    expanded = gl.convert_layout(expanded, gl.DotOperandLayout(1, mma, 8))
    return cdna5.scaled_upcast(packed, expanded, gl.bfloat16, axis=0)


@g.jit
def expert_tile(
    X,
    Sorted,
    W,
    Scale,
    Out,
    block,
    expert,
    tile,
    FIRST: gl.constexpr,
    IN_LAUNCH: gl.constexpr = False,
    READ_MODE: gl.constexpr = 0,
    PARTITIONS: gl.constexpr = 1,
    SPLIT=None,
    SCOPED_X: gl.constexpr = False,
    SCOPED_OUTPUT: gl.constexpr = True,
    WEIGHT_CACHE: gl.constexpr = "",
    WAIT_FLAG=None,
    WAIT_TAG=0,
):
    """Compute one routed M16/N64 tile; K4096, BF16 input/output, native FP4 weights."""
    if IN_LAUNCH and not FIRST and READ_MODE == 1:
        gl.inline_asm_elementwise(
            "s_wait_loadcnt 0\nglobal_inv scope:SCOPE_DEV\nv_mov_b32 $0,0",
            "=v,~{memory}",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    gl.static_assert(gl.num_warps() == 4)
    mma: gl.constexpr = AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        instr_shape=[16, 16, 32],
        rank=2,
    )
    al: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([16, 1], [4, 8], [1, 4], [0, 1])
    sl: gl.constexpr = gl.BlockedLayout([4, 1], [1, 32], [1, 4], [0, 1])
    rows = gl.arange(0, 16, layout=gl.SliceLayout(1, al))
    ak = gl.arange(0, 128, layout=gl.SliceLayout(0, al))
    slots = gl.load(
        Sorted + block * 16 + rows, cache_modifier=".cg" if SCOPED_X else ""
    )
    input_rows = slots // 8 if FIRST else slots
    ap = input_rows[:, None] * 4096 + ak[None, :]
    mask = (slots[:, None] < 64).broadcast_to((16, 128))
    bk = gl.arange(0, 64, layout=gl.SliceLayout(1, bl))
    bn = gl.arange(0, 64, layout=gl.SliceLayout(0, bl)) + tile * 64
    sk = gl.arange(0, 4, layout=gl.SliceLayout(1, sl))
    sn = gl.arange(0, 64, layout=gl.SliceLayout(0, sl)) + tile * 64
    N: gl.constexpr = 8192 if FIRST else 4096
    bp = expert.to(gl.int64) * N * 2048 + bn[None, :] * 2048 + bk[:, None]
    sp = expert.to(gl.int64) * N * 128 + sn[None, :] * 128 + sk[:, None]
    if PARTITIONS > 1:
        gl.static_assert(not FIRST)
        ap += SPLIT * (4096 // PARTITIONS)
        bp += SPLIT * (2048 // PARTITIONS)
        sp += SPLIT * (128 // PARTITIONS)
    ash: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [16, 128], [1, 0]
    )
    bsh: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [64, 64], [0, 1]
    )
    ssh: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 1, [0, 1])
    sa = gl.allocate_shared_memory(gl.bfloat16, [2, 16, 128], ash)
    sb = gl.allocate_shared_memory(gl.uint8, [2, 64, 64], bsh)
    ss = gl.allocate_shared_memory(gl.uint8, [2, 4, 64], ssh)
    if FIRST:
        su = gl.allocate_shared_memory(gl.uint8, [2, 64, 64], bsh)
        st = gl.allocate_shared_memory(gl.uint8, [2, 4, 64], ssh)
    ready = gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    if WAIT_FLAG is not None:
        gl.static_assert(not FIRST and READ_MODE == 2 and PARTITIONS == 1)
        gfx1250.async_copy.global_to_shared(
            sb.index(0), W + bp, cache_modifier=WEIGHT_CACHE
        )
        gfx1250.async_copy.global_to_shared(ss.index(0), Scale + sp)
        gfx1250.async_copy.commit_group()
        seen, polls = scoped.wait_flag(
            WAIT_FLAG, WAIT_TAG, 10_000_000, MAX_SPINS=1_048_576
        )
        ready = seen == WAIT_TAG
    if WAIT_FLAG is None or ready:
        if WAIT_FLAG is None:
            _prefetch(
                X,
                W,
                Scale,
                ap,
                bp,
                sp,
                0,
                0,
                sa,
                sb,
                ss,
                mask,
                IN_LAUNCH and (not FIRST or SCOPED_X),
                2 if FIRST and SCOPED_X else READ_MODE,
                WEIGHT_CACHE,
            )
        else:
            gfx1250.async_copy.global_to_shared(
                sa.index(0), X + ap, mask, 0, cache_modifier=".cg"
            )
        if FIRST:
            gfx1250.async_copy.global_to_shared(
                su.index(0), W + bp + 4096 * 2048, cache_modifier=WEIGHT_CACHE
            )
            gfx1250.async_copy.global_to_shared(st.index(0), Scale + sp + 4096 * 128)
        gfx1250.async_copy.commit_group()
        acc = gl.full((16, 64), 0, gl.float32, mma)
        if FIRST:
            up = gl.full((16, 64), 0, gl.float32, mma)
        for step in range(32 // PARTITIONS):
            bank = step % 2
            gfx1250.async_copy.wait_group(0)
            gl.barrier()
            if step + 1 < 32 // PARTITIONS:
                _prefetch(
                    X,
                    W,
                    Scale,
                    ap,
                    bp,
                    sp,
                    step + 1,
                    1 - bank,
                    sa,
                    sb,
                    ss,
                    mask,
                    IN_LAUNCH and (not FIRST or SCOPED_X),
                    2 if FIRST and SCOPED_X else READ_MODE,
                    WEIGHT_CACHE,
                )
                if FIRST:
                    gfx1250.async_copy.global_to_shared(
                        su.index(1 - bank),
                        W + bp + (step + 1) * 64 + 4096 * 2048,
                        cache_modifier=WEIGHT_CACHE,
                    )
                    gfx1250.async_copy.global_to_shared(
                        st.index(1 - bank), Scale + sp + (step + 1) * 4 + 4096 * 128
                    )
                gfx1250.async_copy.commit_group()
            a = sa.index(bank).load(gl.DotOperandLayout(0, mma, 8))
            b = _weight(sb, ss, bank, mma)
            acc = gfx1250.wmma(a, b, acc)
            if FIRST:
                u = _weight(su, st, bank, mma)
                up = gfx1250.wmma(a, u, up)
            # All waves finish reading this bank before it is reused two steps later.
            gl.barrier()
        gfx1250.async_copy.wait_group(0)
        if FIRST:
            acc = acc / (1.0 + gl.exp(-acc)) * up
        mr = gl.arange(0, 16, layout=gl.SliceLayout(1, mma))
        nc = gl.arange(0, 64, layout=gl.SliceLayout(0, mma)) + tile * 64
        output_slots = gl.load(
            Sorted + block * 16 + mr, cache_modifier=".cg" if SCOPED_X else ""
        )
        if PARTITIONS > 1:
            store_tile_words(
                Out.to(gl.pointer_type(gl.int32))
                + output_slots[:, None] * 4096
                + nc[None, :],
                acc.to(gl.int32, bitcast=True),
                output_slots[:, None] < 64,
                16 * 64,
            )
        elif IN_LAUNCH and SCOPED_OUTPUT:
            packed_layout: gl.constexpr = gl.BlockedLayout(
                [1, 8], [2, 16], [4, 1], [1, 0]
            )
            rounded = gl.convert_layout(acc.to(gl.bfloat16), packed_layout)
            lo, hi = gl.split(rounded.reshape((16, 32, 2)))
            bits = lo.to(gl.uint16, bitcast=True).to(gl.uint32) | (
                hi.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
            )
            rr = gl.arange(0, 16, layout=gl.SliceLayout(1, bits.type.layout))
            cc = gl.arange(0, 32, layout=gl.SliceLayout(0, bits.type.layout))
            hit = gl.load(
                Sorted + block * 16 + rr, cache_modifier=".cg" if SCOPED_X else ""
            )
            scoped.store_words(
                Out.to(gl.pointer_type(gl.int32))
                + hit[:, None] * 2048
                + tile * 32
                + cc[None, :],
                bits.to(gl.int32),
                hit[:, None] < 64,
            )
        else:
            gl.store(
                Out + output_slots[:, None] * 4096 + nc[None, :],
                acc.to(gl.bfloat16),
                output_slots[:, None] < 64,
            )
    else:
        gfx1250.async_copy.wait_group(0)
    return ready


@g.jit
def _launch(X, Sorted, Experts, NumPost, W, Scale, Out, FIRST: gl.constexpr):
    task = gl.program_id(0)
    if task < gl.load(NumPost) // 16 * 64:
        block = task // 64
        expert = gl.load(Experts + block)
        if expert >= 0:
            expert_tile(X, Sorted, W, Scale, Out, block, expert, task % 64, FIRST)


def experts(x, sorted_ids, expert_ids, num_post, w13, s13, w2, s2, inter, output):
    """Two non-persistent launches, gfx1250/Triton3.9 only; caller owns intermediates."""
    grid = (expert_ids.numel() * 64,)
    first = _launch[grid](
        x, sorted_ids, expert_ids, num_post, w13, s13, inter, True, num_warps=4
    )
    second = _launch[grid](
        inter, sorted_ids, expert_ids, num_post, w2, s2, output, False, num_warps=4
    )
    return first, second

"""TLX BF16-weight expert tile for the two MoE layers Quark leaves unquantized (0 and 61).

Same task, slot and output contract as tlx_expert_wide.expert_tile_wide with MX off: one CTA
computes 64 output columns of one expert for one 16-slot block. Only the B side differs: the
weights are bf16 [N, 4096] row-major per expert (w13 [E, 8192, 4096], w2 [E, 4096, 4096]),
streamed in K_STEP slices through DEPTH LDS banks and fed to the WMMA as four-or-fewer K128
subtiles, so no upcast and no scales.
"""

import triton
import triton.language as tl
import utlx_plugin as tlx

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
    MMA,
    async_agent,
)

# 8 bf16 (16 B) per thread along K, the contiguous dimension of a [K, N] view of [N, K] weights.
BL16 = tl.constexpr(BlockedEncoding([8, 1], [8, 4], [1, 4], [0, 1]))


@triton.jit
def _load_a(X, ap, step, bank, sa, mask, FIRST: tl.constexpr, CROSS_LAUNCH_A: tl.constexpr,
            K_STEP: tl.constexpr):
    zero = tlx.require_layout(tl.full((16, K_STEP), 0.0, tl.bfloat16), AL)
    if FIRST or CROSS_LAUNCH_A:
        tlx.async_load(X + ap + step * K_STEP, tlx.local_view(sa, bank), mask, zero)
    else:
        async_agent(X + ap + step * K_STEP, tlx.local_view(sa, bank), mask, zero)


@triton.jit
def _issue(X, W, ap, bp, step, bank, sa, sb, su, mask, FIRST: tl.constexpr,
           CROSS_LAUNCH_A: tl.constexpr, K_STEP: tl.constexpr):
    K_DIM: tl.constexpr = 4096
    _load_a(X, ap, step, bank, sa, mask, FIRST, CROSS_LAUNCH_A, K_STEP)
    tlx.async_load(W + bp + step * K_STEP, tlx.local_view(sb, bank))
    if FIRST:
        # up rows follow the 4096 gate rows of w13
        tlx.async_load(W + bp + step * K_STEP + 4096 * K_DIM, tlx.local_view(su, bank))
    tlx.async_load_commit_group()


@triton.jit
def expert_tile_bf16(
    X,
    Sorted,
    W,
    Out,
    block,
    expert,
    tile,
    FIRST: tl.constexpr,
    CROSS_LAUNCH_A: tl.constexpr = False,
    OUTPUT_AGENT: tl.constexpr = False,
    DEPTH: tl.constexpr = 2,
    K_STEP: tl.constexpr = 256,
    SLOTS=None,
):
    K_DIM: tl.constexpr = 4096
    tl.static_assert(K_DIM % K_STEP == 0)
    tl.static_assert(K_STEP % 128 == 0)
    NSTEP: tl.constexpr = K_DIM // K_STEP
    SUBS: tl.constexpr = K_STEP // 128
    tl.static_assert(NSTEP >= DEPTH)
    row = tl.arange(0, 16)
    if SLOTS is None:
        slots = tl.load(Sorted + block * 16 + row)
    else:
        slots = SLOTS
    input_rows = slots // 8 if FIRST else slots
    ak = tl.arange(0, K_STEP)
    ap = tlx.require_layout(input_rows[:, None] * K_DIM + ak[None, :], AL)
    mask = tlx.require_layout(tl.broadcast_to(slots[:, None] < 64, (16, K_STEP)), AL)
    N: tl.constexpr = 8192 if FIRST else 4096
    bk = tl.arange(0, K_STEP)
    bn = tl.arange(0, 64) + tile * 64
    bp = tlx.require_layout(
        expert.to(tl.int64) * N * K_DIM + bn[None, :] * K_DIM + bk[:, None], BL16
    )
    ash: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(K_STEP, 8)], [16, K_STEP], [1, 0]
    )
    bsh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(K_STEP, 8)], [K_STEP, 64], [0, 1]
    )
    sa = tlx.local_alloc((16, K_STEP), tl.bfloat16, DEPTH, layout=ash)
    sb = tlx.local_alloc((K_STEP, 64), tl.bfloat16, DEPTH, layout=bsh)
    su = sb
    if FIRST:
        su = tlx.local_alloc((K_STEP, 64), tl.bfloat16, DEPTH, layout=bsh)
    for init_step in tl.static_range(DEPTH - 1):
        _issue(X, W, ap, bp, init_step, init_step, sa, sb, su, mask, FIRST, CROSS_LAUNCH_A,
               K_STEP)
    acc = fixed(tl.zeros((16, 64), tl.float32), MMA)
    if FIRST:
        up = fixed(tl.zeros((16, 64), tl.float32), MMA)
    for step in range(NSTEP):
        bank = step % DEPTH
        if step + DEPTH - 1 < NSTEP:
            _issue(X, W, ap, bp, step + DEPTH - 1, (step + DEPTH - 1) % DEPTH, sa, sb, su, mask,
                   FIRST, CROSS_LAUNCH_A, K_STEP)
            tlx.async_load_wait_group(DEPTH - 1)
        else:
            tlx.async_load_wait_group(0)
        tlx.workgroup_barrier()
        for sub in tl.static_range(SUBS):
            a = tlx.local_load(
                tlx.local_slice(tlx.local_view(sa, bank), [0, sub * 128], [16, 128]),
                layout=A_LAYOUT,
            )
            b = tlx.local_load(
                tlx.local_slice(tlx.local_view(sb, bank), [sub * 128, 0], [128, 64]),
                layout=B_EXPANDED,
            )
            acc = tl.dot(a, b, acc)
            if FIRST:
                u = tlx.local_load(
                    tlx.local_slice(tlx.local_view(su, bank), [sub * 128, 0], [128, 64]),
                    layout=B_EXPANDED,
                )
                up = tl.dot(a, u, up)
        tlx.workgroup_barrier()
    tlx.async_load_wait_group(0)
    if FIRST:
        acc = acc / (1.0 + tl.exp(-acc)) * up
    col = tl.arange(0, 64) + tile * 64
    output_slots = tl.load(Sorted + block * 16 + row)
    tl.store(
        Out + output_slots[:, None] * 4096 + col[None, :],
        acc.to(tl.bfloat16),
        (output_slots < 64)[:, None],
        cache_modifier=".cg" if OUTPUT_AGENT else "",
    )
    return True

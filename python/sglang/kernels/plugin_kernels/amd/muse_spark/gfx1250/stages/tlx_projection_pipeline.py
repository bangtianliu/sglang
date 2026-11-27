"""T1 bounded projection-layout pass; no stage selected before validation."""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_guard
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    WMMA,
    WMMA_A,
    WMMA_B,
    BlockedEncoding,
)

LOAD = tl.constexpr(BlockedEncoding([1, 8], [2, 16], [2, 1], [1, 0]))


@triton.jit
def issue_async(X, W, sa, sb, tile, step, bank, K: tl.constexpr):
    r = tl.arange(0, 16)
    k = step * 512 + tl.arange(0, 512)
    ap = tlx.require_layout(X + r[:, None] * K + k[None, :], LOAD)
    bp = tlx.require_layout(W + (tile * 16 + r[:, None]) * K + k[None, :], LOAD)
    ap = tl.max_contiguous(tl.multiple_of(ap, (1, 8)), (1, 8))
    bp = tl.max_contiguous(tl.multiple_of(bp, (1, 8)), (1, 8))
    mask = tlx.require_layout(tl.broadcast_to(r[:, None] < 8, (16, 512)), LOAD)
    zero = tlx.require_layout(tl.full((16, 512), 0, tl.bfloat16), LOAD)
    tlx.async_load(ap, tlx.local_view(sa, bank), mask, zero)
    tlx.async_load(bp, tlx.local_view(sb, bank))
    tlx.async_load_commit_group()


@triton.jit
def project(
    X,
    W,
    P,
    Status,
    PINNED: tl.constexpr = True,
    POST_BARRIER: tl.constexpr = True,
    Failure=None,
    SKIP: tl.constexpr = -1,
    Seed=None,
    Packed=None,
    N: tl.constexpr = 8192,
    K: tl.constexpr = 4096,
    STAGE: tl.constexpr = 6,
    SK: tl.constexpr = 4,
    ROUND_OUTPUT: tl.constexpr = True,
    DEPTH: tl.constexpr = 4,
    COPY: tl.constexpr = "tdm",
):
    tl.static_assert(SK == 4 or SK == 8)
    tl.static_assert(K >= 2048 and K % (SK * 512) == 0 and N % 16 == 0)
    tile = tl.program_id(0)
    tlx_guard.delay(Seed, tile, 0)
    allowed = tlx_guard.enabled(Failure, STAGE)
    ok = allowed & (tile != SKIP)
    if ok:
        sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
            [(512, 8)], [16, 512], [1, 0]
        )
        sa = tlx.local_alloc((16, 512), tl.bfloat16, DEPTH, layout=sh)
        sb = tlx.local_alloc((16, 512), tl.bfloat16, DEPTH, layout=sh)
        ad, bd = None, None
        if COPY == "tdm":
            ad = tlx.make_tensor_descriptor(None, X, [8, K], [K, 1], [16, 512])
            bd = tlx.make_tensor_descriptor(
                None, W + tile * 16 * K, [16, K], [K, 1], [16, 512]
            )
        for step in tl.static_range(min(DEPTH - 1, K // 512)):
            if COPY == "tdm":
                tlx.tdm_async_load(ad, tlx.local_view(sa, step), [0, step * 512])
                tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
            else:
                issue_async(X, W, sa, sb, tile, step, step, K)
        parts = ()
        acc = tl.zeros((16, 16), tl.float32)
        if PINNED:
            acc = tlx.require_layout(acc, WMMA)
        for step in tl.static_range(K // 512):
            if step + DEPTH - 1 < K // 512:
                future = step + DEPTH - 1
                if COPY == "tdm":
                    tlx.tdm_async_load(
                        ad, tlx.local_view(sa, future % DEPTH), [0, future * 512]
                    )
                    tlx.tdm_async_load(
                        bd, tlx.local_view(sb, future % DEPTH), [0, future * 512]
                    )
                else:
                    issue_async(X, W, sa, sb, tile, future, future % DEPTH, K)
            if COPY == "tdm":
                tlx.tdm_async_wait(min(DEPTH - 1, K // 512 - step - 1) * 2)
            else:
                tlx.async_load_wait_group(min(DEPTH - 1, K // 512 - step - 1))
                tlx.workgroup_barrier()
            aa = tlx.local_load(tlx.local_view(sa, step % DEPTH))
            if PINNED:
                aa = tlx.require_layout(aa, WMMA_A)
            bb = tlx.local_load(tlx.local_trans(tlx.local_view(sb, step % DEPTH)))
            if PINNED:
                bb = tlx.require_layout(bb, WMMA_B)
            # All waves have consumed this bank before a later TDM overwrite.
            if not POST_BARRIER:
                tlx.workgroup_barrier()
            if step % (K // SK // 512) == 0:
                acc = tl.zeros((16, 16), tl.float32)
                if PINNED:
                    acc = tlx.require_layout(acc, WMMA)
            acc = tl.dot(aa, bb, acc)
            if POST_BARRIER:
                tlx.workgroup_barrier()
            if (step + 1) % (K // SK // 512) == 0:
                parts += (acc,)
        if COPY == "tdm":
            tlx.tdm_async_wait(0)
        else:
            tlx.async_load_wait_group(0)
        tlx_guard.delay(Seed, tile, 1)
        if SK == 8:
            value = ((parts[0] + parts[4]) + (parts[2] + parts[6])) + (
                (parts[1] + parts[5]) + (parts[3] + parts[7])
            )
        else:
            value = (parts[0] + parts[2]) + (parts[1] + parts[3])
        if ROUND_OUTPUT:
            value = value.to(tl.bfloat16).to(tl.float32)
        r = tl.arange(0, 16)
        n = tile * 16 + tl.arange(0, 16)
        tl.store(
            P + r[:, None] * N + n[None, :],
            value.to(tl.int32, bitcast=True),
            r[:, None] < 8,
        )
        if Packed is not None:
            tl.store(
                Packed + r[:, None] * N + n[None, :],
                value.to(tl.bfloat16),
                r[:, None] < 8,
            )
    tlx_guard.delay(Seed, tile, 2)
    tl.store(Status + tile, ok)
    if allowed:
        tlx_guard.finish(Failure, ok, tile, STAGE)

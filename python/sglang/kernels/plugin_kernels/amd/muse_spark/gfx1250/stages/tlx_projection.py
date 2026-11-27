"""TLX projection with selectable copy pipeline and a fixed split-K tree."""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard,
    tlx_memory,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    WMMA,
    WMMA_A,
    WMMA4,
    WMMA4_A,
    WMMA4_B,
    WMMA_B,
    BlockedEncoding,
)

# Launchers pass EARLY=EARLY_ISSUE: prologue loads before the failure-word load.
EARLY_ISSUE = True
# Activation bank copy: "tdm" (X and W through TDM) or "hybrid" (X through the
# vector async path, W through TDM). Same LDS bytes either way.
COPY_MODE = "tdm"
# TDM bank depth of the S1 (project_vg), S3 and S4 projections.
DEPTH_S1 = 4
DEPTH_S3 = 2
DEPTH_S4 = 4
# Columns per CTA (16 or 32) of the S1 (project_vg) / S3 / S4 / S6 projections.
BN_S1 = 32
BN_S3 = 32
BN_S4 = 32
BN_S6 = 64

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
def issue_x(X, sa, step, bank, K: tl.constexpr):
    """X bank through the vector-memory async path (the TDM engine only carries W).
    Same LDS image as the TDM copy: rows 0-7 = X, rows 8-15 = 0."""
    r = tl.arange(0, 16)
    k = step * 512 + tl.arange(0, 512)
    ap = tlx.require_layout(X + r[:, None] * K + k[None, :], LOAD)
    ap = tl.max_contiguous(tl.multiple_of(ap, (1, 8)), (1, 8))
    mask = tlx.require_layout(tl.broadcast_to(r[:, None] < 8, (16, 512)), LOAD)
    zero = tlx.require_layout(tl.full((16, 512), 0, tl.bfloat16), LOAD)
    tlx.async_load(ap, tlx.local_view(sa, bank), mask, zero)
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
    EARLY: tl.constexpr = False,
    BN: tl.constexpr = 16,
):
    project_body(X, W, P, Status, PINNED, POST_BARRIER, Failure, SKIP, Seed, Packed,
                 N, K, STAGE, SK, ROUND_OUTPUT, DEPTH, COPY, False, EARLY, BN)


@triton.jit
def project_body(
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
    SCOPED: tl.constexpr = False,
    EARLY: tl.constexpr = False,
    BN: tl.constexpr = 16,
):
    """One BN-column tile (BN 16: the two warps replicate one 16x16 WMMA tile; BN 32:
    each warp owns 16 columns, same per-column K order). SCOPED: P goes out SCOPE_DEV
    for an in-kernel reader. Returns ok. Status stays per 16-column tile."""
    tl.static_assert(SK == 4 or SK == 8)
    tl.static_assert(BN == 16 or ((BN == 32 or BN == 64) and COPY != "async"))
    tl.static_assert(K >= 2048 and K % (SK * 512) == 0 and N % 16 == 0)
    tile = tl.program_id(0)
    tlx_guard.delay(Seed, tile, 0)
    if EARLY:
        # issue the first weight/activation banks before the failure-word round trip
        sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
            [(512, 8)], [16, 512], [1, 0]
        )
        shb: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
            [(512, 8)], [BN, 512], [1, 0]
        )
        sa = tlx.local_alloc((16, 512), tl.bfloat16, DEPTH, layout=sh)
        sb = tlx.local_alloc((BN, 512), tl.bfloat16, DEPTH, layout=shb)
        ad, bd = None, None
        if COPY == "tdm":
            ad = tlx.make_tensor_descriptor(None, X, [8, K], [K, 1], [16, 512])
        if COPY == "tdm" or COPY == "hybrid":
            bd = tlx.make_tensor_descriptor(
                None, W + tile * BN * K, [BN, K], [K, 1], [BN, 512]
            )
        for step in tl.static_range(min(DEPTH - 1, K // 512)):
            if COPY == "tdm":
                tlx.tdm_async_load(ad, tlx.local_view(sa, step), [0, step * 512])
                tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
            elif COPY == "hybrid":
                issue_x(X, sa, step, step, K)
                tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
            else:
                issue_async(X, W, sa, sb, tile, step, step, K)
    allowed = tlx_guard.enabled(Failure, STAGE)
    ok = allowed & (tile != SKIP)
    value = tl.zeros((16, BN), tl.float32)
    if PINNED:
        if BN == 64:
            value = tlx.require_layout(value, WMMA4)
        else:
            value = tlx.require_layout(value, WMMA)
    if ok:
        if not EARLY:
            sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
                [(512, 8)], [16, 512], [1, 0]
            )
            shb: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
                [(512, 8)], [BN, 512], [1, 0]
            )
            sa = tlx.local_alloc((16, 512), tl.bfloat16, DEPTH, layout=sh)
            sb = tlx.local_alloc((BN, 512), tl.bfloat16, DEPTH, layout=shb)
            ad, bd = None, None
            if COPY == "tdm":
                ad = tlx.make_tensor_descriptor(None, X, [8, K], [K, 1], [16, 512])
            if COPY == "tdm" or COPY == "hybrid":
                bd = tlx.make_tensor_descriptor(
                    None, W + tile * BN * K, [BN, K], [K, 1], [BN, 512]
                )
            for step in tl.static_range(min(DEPTH - 1, K // 512)):
                if COPY == "tdm":
                    tlx.tdm_async_load(ad, tlx.local_view(sa, step), [0, step * 512])
                    tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
                elif COPY == "hybrid":
                    issue_x(X, sa, step, step, K)
                    tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
                else:
                    issue_async(X, W, sa, sb, tile, step, step, K)
        parts = ()
        acc = tl.zeros((16, BN), tl.float32)
        if PINNED:
            if BN == 64:
                acc = tlx.require_layout(acc, WMMA4)
            else:
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
                elif COPY == "hybrid":
                    issue_x(X, sa, future, future % DEPTH, K)
                    tlx.tdm_async_load(
                        bd, tlx.local_view(sb, future % DEPTH), [0, future * 512]
                    )
                else:
                    issue_async(X, W, sa, sb, tile, future, future % DEPTH, K)
            if COPY == "tdm":
                tlx.tdm_async_wait(min(DEPTH - 1, K // 512 - step - 1) * 2)
            elif COPY == "hybrid":
                tlx.tdm_async_wait(min(DEPTH - 1, K // 512 - step - 1))
                tlx.async_load_wait_group(min(DEPTH - 1, K // 512 - step - 1))
                tlx.workgroup_barrier()
            else:
                tlx.async_load_wait_group(min(DEPTH - 1, K // 512 - step - 1))
                tlx.workgroup_barrier()
            aa = tlx.local_load(tlx.local_view(sa, step % DEPTH))
            if PINNED:
                if BN == 64:
                    aa = tlx.require_layout(aa, WMMA4_A)
                else:
                    aa = tlx.require_layout(aa, WMMA_A)
            bb = tlx.local_load(tlx.local_trans(tlx.local_view(sb, step % DEPTH)))
            if PINNED:
                if BN == 64:
                    bb = tlx.require_layout(bb, WMMA4_B)
                else:
                    bb = tlx.require_layout(bb, WMMA_B)
            # All waves have consumed this bank before a later TDM overwrite.
            if not POST_BARRIER:
                tlx.workgroup_barrier()
            if step % (K // SK // 512) == 0:
                acc = tl.zeros((16, BN), tl.float32)
                if PINNED:
                    if BN == 64:
                        acc = tlx.require_layout(acc, WMMA4)
                    else:
                        acc = tlx.require_layout(acc, WMMA)
            acc = tl.dot(aa, bb, acc)
            if POST_BARRIER:
                tlx.workgroup_barrier()
            if (step + 1) % (K // SK // 512) == 0:
                parts += (acc,)
        if COPY == "tdm":
            tlx.tdm_async_wait(0)
        elif COPY == "hybrid":
            tlx.tdm_async_wait(0)
            tlx.async_load_wait_group(0)
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
        n = tile * BN + tl.arange(0, BN)
        if SCOPED == 2:
            # publish with returning device-scope atomics; the caller makes its
            # counter atomic depend on the returned words
            gone = tl.atomic_xchg(
                P + r[:, None] * N + n[None, :],
                value.to(tl.int32, bitcast=True),
                mask=tl.broadcast_to(r[:, None] < 8, (16, BN)),
                sem="relaxed",
                scope="gpu",
            )
            ok = ok & (tl.sum(tl.sum(gone * 0, 1), 0) == 0)
        elif SCOPED:
            tlx_memory.store_words(
                P + r[:, None] * N + n[None, :],
                value.to(tl.int32, bitcast=True),
                tl.broadcast_to(r[:, None] < 8, (16, BN)),
            )
        else:
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
    else:
        if EARLY:
            if COPY == "tdm":
                tlx.tdm_async_wait(0)
            elif COPY == "hybrid":
                tlx.tdm_async_wait(0)
                tlx.async_load_wait_group(0)
            else:
                tlx.async_load_wait_group(0)
    tlx_guard.delay(Seed, tile, 2)
    if BN == 16:
        tl.store(Status + tile, ok)
    else:
        tl.store(Status + tile * (BN // 16) + tl.arange(0, BN // 16), ok)
    if allowed:
        tlx_guard.finish(Failure, ok, tile, STAGE)
    return ok, value

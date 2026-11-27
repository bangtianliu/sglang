"""T10 load-only projection grid/pipeline search; no production import."""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_guard
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    BlockedEncoding,
)

LOAD = tl.constexpr(BlockedEncoding([1, 8], [2, 16], [2, 1], [1, 0]))


@triton.jit
def issue(
    X,
    W,
    sa,
    sb,
    ad,
    bd,
    tile,
    split,
    step,
    bank,
    K: tl.constexpr,
    KC: tl.constexpr,
    BK: tl.constexpr,
    COPY: tl.constexpr,
):
    if COPY == "tdm":
        tlx.tdm_async_load(ad, tlx.local_view(sa, bank), [0, step * BK])
        tlx.tdm_async_load(bd, tlx.local_view(sb, bank), [0, step * BK])
    else:
        r = tl.arange(0, 16)
        k = split * KC + step * BK + tl.arange(0, BK)
        ap = tlx.require_layout(X + r[:, None] * K + k[None, :], LOAD)
        bp = tlx.require_layout(W + (tile * 16 + r[:, None]) * K + k[None, :], LOAD)
        ap = tl.max_contiguous(tl.multiple_of(ap, (1, 8)), (1, 8))
        bp = tl.max_contiguous(tl.multiple_of(bp, (1, 8)), (1, 8))
        mask = tlx.require_layout(tl.broadcast_to(r[:, None] < 8, (16, BK)), LOAD)
        zero = tlx.require_layout(tl.full((16, BK), 0, tl.bfloat16), LOAD)
        tlx.async_load(ap, tlx.local_view(sa, bank), mask, zero)
        tlx.async_load(bp, tlx.local_view(sb, bank))
        tlx.async_load_commit_group()


@triton.jit
def sampled(buffer):
    view = tlx.local_slice(buffer, [0, 0], [1, 2])
    value = tlx.local_load(view).to(tl.uint16, bitcast=True).to(tl.uint32)
    return tl.sum(tl.sum(value, 1), 0)


@triton.jit
def load_project(
    X,
    W,
    Checksum,
    Status,
    Failure=None,
    N: tl.constexpr = 4608,
    K: tl.constexpr = 8192,
    PSPLIT: tl.constexpr = 1,
    DEPTH: tl.constexpr = 4,
    BK: tl.constexpr = 512,
    COPY: tl.constexpr = "tdm",
    STAGE: tl.constexpr = 1,
):
    tl.static_assert(K % PSPLIT == 0 and (K // PSPLIT) % BK == 0)
    pid = tl.program_id(0)
    tile, split = pid // PSPLIT, pid % PSPLIT
    KC: tl.constexpr = K // PSPLIT
    STEPS: tl.constexpr = KC // BK
    ok = tlx_guard.enabled(Failure, STAGE)
    if ok:
        sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
            [(BK, 8)], [16, BK], [1, 0]
        )
        sa = tlx.local_alloc(
            (16, BK),
            tl.bfloat16,
            DEPTH,
            layout=None if COPY == "tdm" and BK == 1024 else sh,
        )
        sb = tlx.local_alloc(
            (16, BK),
            tl.bfloat16,
            DEPTH,
            layout=None if COPY == "tdm" and BK == 1024 else sh,
        )
        ad, bd = None, None
        if COPY == "tdm":
            ad = tlx.make_tensor_descriptor(
                None, X + split * KC, [8, KC], [K, 1], [16, BK]
            )
            bd = tlx.make_tensor_descriptor(
                None, W + tile * 16 * K + split * KC, [16, KC], [K, 1], [16, BK]
            )
        for init in tl.static_range(min(DEPTH - 1, STEPS)):
            if COPY == "tdm":
                tlx.tdm_async_load(ad, tlx.local_view(sa, init), [0, init * BK])
                tlx.tdm_async_load(bd, tlx.local_view(sb, init), [0, init * BK])
            else:
                issue(X, W, sa, sb, ad, bd, tile, split, init, init, K, KC, BK, COPY)
        crc = tl.full((), 0, tl.uint32)
        for step in tl.static_range(STEPS):
            if step + DEPTH - 1 < STEPS:
                future = step + DEPTH - 1
                if COPY == "tdm":
                    tlx.tdm_async_load(
                        ad, tlx.local_view(sa, future % DEPTH), [0, future * BK]
                    )
                    tlx.tdm_async_load(
                        bd, tlx.local_view(sb, future % DEPTH), [0, future * BK]
                    )
                else:
                    issue(
                        X,
                        W,
                        sa,
                        sb,
                        ad,
                        bd,
                        tile,
                        split,
                        future,
                        future % DEPTH,
                        K,
                        KC,
                        BK,
                        COPY,
                    )
            if COPY == "tdm":
                tlx.tdm_async_wait(min(DEPTH - 1, STEPS - step - 1) * 2)
            else:
                tlx.async_load_wait_group(min(DEPTH - 1, STEPS - step - 1))
                tlx.workgroup_barrier()
            crc += sampled(tlx.local_view(sa, step % DEPTH)) + sampled(
                tlx.local_view(sb, step % DEPTH)
            )
            tlx.workgroup_barrier()
        if COPY == "tdm":
            tlx.tdm_async_wait(0)
        else:
            tlx.async_load_wait_group(0)
        tl.store(Checksum + pid, crc)
    tl.store(Status + pid, ok)
    tlx_guard.finish(Failure, ok, pid, STAGE)

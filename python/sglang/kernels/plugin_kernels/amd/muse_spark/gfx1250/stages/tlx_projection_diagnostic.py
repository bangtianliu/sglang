"""T10 same-body projection decomposition; never selected by the chain."""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_guard
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    WMMA,
    WMMA_A,
    WMMA_B,
)


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
    MODE: tl.constexpr = "full",
    Checksum=None,
):
    tl.static_assert(COPY == "tdm")
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
        if MODE == "compute":
            row = tl.arange(0, 16)
            column = tl.arange(0, 128)
            for init_bank in tl.static_range(DEPTH):
                for chunk in tl.static_range(4):
                    col = column + chunk * 128
                    av = tl.where(
                        row[:, None] < 8,
                        ((row[:, None] + col[None, :] % 4 + tile) % 8 - 4).to(
                            tl.float32
                        )
                        * 0.0625,
                        0,
                    ).to(tl.bfloat16)
                    bv = (
                        ((row[:, None] * 3 + col[None, :] % 4 + tile) % 16 - 8).to(
                            tl.float32
                        )
                        * 0.03125
                    ).to(tl.bfloat16)
                    tlx.local_store(
                        tlx.local_slice(
                            tlx.local_view(sa, init_bank), [0, chunk * 128], [16, 128]
                        ),
                        av,
                    )
                    tlx.local_store(
                        tlx.local_slice(
                            tlx.local_view(sb, init_bank), [0, chunk * 128], [16, 128]
                        ),
                        bv,
                    )
            tlx.workgroup_barrier()
        else:
            ad = tlx.make_tensor_descriptor(None, X, [8, K], [K, 1], [16, 512])
            bd = tlx.make_tensor_descriptor(
                None, W + tile * 16 * K, [16, K], [K, 1], [16, 512]
            )
            for step in tl.static_range(min(DEPTH - 1, K // 512)):
                tlx.tdm_async_load(ad, tlx.local_view(sa, step), [0, step * 512])
                tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
        if MODE == "load":
            crc = tl.full((), 0, tl.uint32)
        parts = ()
        acc = tl.zeros((16, 16), tl.float32)
        if PINNED:
            acc = tlx.require_layout(acc, WMMA)
        for step in tl.static_range(K // 512):
            if MODE != "compute" and step + DEPTH - 1 < K // 512:
                tlx.tdm_async_load(
                    ad,
                    tlx.local_view(sa, (step + DEPTH - 1) % DEPTH),
                    [0, (step + DEPTH - 1) * 512],
                )
                tlx.tdm_async_load(
                    bd,
                    tlx.local_view(sb, (step + DEPTH - 1) % DEPTH),
                    [0, (step + DEPTH - 1) * 512],
                )
            tlx.tdm_async_wait(min(DEPTH - 1, K // 512 - step - 1) * 2)
            if MODE == "load":
                # TDM still transfers its complete16x512descriptor block.
                # One uint32 from each bank keeps every transaction observable.
                ax = tlx.local_load(
                    tlx.local_slice(tlx.local_view(sa, step % DEPTH), [0, 0], [1, 2])
                )
                bx = tlx.local_load(
                    tlx.local_slice(tlx.local_view(sb, step % DEPTH), [0, 0], [1, 2])
                )
                crc += tl.sum(ax.to(tl.uint16, bitcast=True).to(tl.uint32)) + tl.sum(
                    bx.to(tl.uint16, bitcast=True).to(tl.uint32)
                )
                tlx.workgroup_barrier()
            else:
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
        tlx.tdm_async_wait(0)
        tlx_guard.delay(Seed, tile, 1)
        if MODE == "load":
            tl.store(Checksum + tile, crc)
        else:
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

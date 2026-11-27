"""Experimental S4 projection consuming a per-CTA flag; not chain-selected."""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    WMMA,
    WMMA_A,
    WMMA_B,
    BlockedEncoding,
)

FLAG_LAYOUT = tl.constexpr(BlockedEncoding([1], [32], [2], [0]))


@triton.jit
def flag2(Flags):
    lanes = tlx.require_layout(tl.arange(0, 64), FLAG_LAYOUT)
    saved, value = tl.inline_asm_elementwise(
        "v_mov_b32 $1,0\nv_cmp_ne_u32_e64 $0,0,$3\ns_and_saveexec_b32 $0,$0\nglobal_load_b32 $1,$2,off scope:SCOPE_DEV\ns_wait_loadcnt 0\ns_mov_b32 exec_lo,$0\ns_wait_xcnt 0x0",
        "=&s,=&v,v,v,~{scc}",
        [
            tlx.release_layout(Flags + tl.full((64,), 0, tl.int32)),
            tlx.release_layout((lanes == 0).to(tl.int32)),
        ],
        dtype=(tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    return tl.sum(value, 0)


@triton.jit
def reset(Epoch, Counter, Record):
    tl.store(Counter, 0)
    tl.store(Record + tl.arange(0, 8), 0)
    tl.atomic_add(Epoch, 1, sem="relaxed", scope="gpu")


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
    READY=None,
    EPOCH=None,
    RECORD=None,
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
        sa = tlx.local_alloc((16, 512), tl.bfloat16, 4, layout=sh)
        sb = tlx.local_alloc((16, 512), tl.bfloat16, 4, layout=sh)
        ad = tlx.make_tensor_descriptor(None, X, [8, K], [K, 1], [16, 512])
        bd = tlx.make_tensor_descriptor(
            None, W + tile * 16 * K, [16, K], [K, 1], [16, 512]
        )
        for step in tl.static_range(3):
            tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
        expected_tag = tl.load(EPOCH)
        observed = flag2(READY + tile * 32)
        polls = 0
        while (observed != expected_tag) & (polls < 4096):
            tl.inline_asm_elementwise(
                "s_sleep 1\nv_mov_b32 $0,0",
                "=v",
                [],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            observed = flag2(READY + tile * 32)
            polls += 1
        ok = observed == expected_tag
        if not ok:
            owner = tl.atomic_cas(RECORD, 0, 1, sem="relaxed", scope="gpu")
            if owner == 0:
                tl.store(RECORD + 1, tile)
                tl.store(RECORD + 2, expected_tag)
                tl.store(RECORD + 3, observed)
                tl.store(RECORD + 4, polls)
        # Missing-producer runs leave output untouched and publish failure.
        for start_step in tl.static_range(3):
            if ok:
                tlx.tdm_async_load(
                    ad,
                    tlx.local_view(sa, start_step),
                    [0, start_step * 512],
                    cache_modifier=".cg",
                    scope="gpu",
                )
        if ok:
            # All initial B-then-A copies must finish before paired ring waits.
            tlx.tdm_async_wait(0)
            parts = ()
            acc = tl.zeros((16, 16), tl.float32)
            if PINNED:
                acc = tlx.require_layout(acc, WMMA)
            for step in tl.static_range(K // 512):
                if step + 3 < K // 512:
                    tlx.tdm_async_load(
                        ad,
                        tlx.local_view(sa, (step + 3) % 4),
                        [0, (step + 3) * 512],
                        cache_modifier=".cg",
                        scope="gpu",
                    )
                    tlx.tdm_async_load(
                        bd, tlx.local_view(sb, (step + 3) % 4), [0, (step + 3) * 512]
                    )
                tlx.tdm_async_wait(min(3, K // 512 - step - 1) * 2)
                aa = tlx.local_load(tlx.local_view(sa, step % 4))
                if PINNED:
                    aa = tlx.require_layout(aa, WMMA_A)
                bb = tlx.local_load(tlx.local_trans(tlx.local_view(sb, step % 4)))
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
    tlx.tdm_async_wait(0)
    tlx_guard.delay(Seed, tile, 2)
    tl.store(Status + tile, ok)
    if allowed:
        tlx_guard.finish(Failure, ok, tile, STAGE)

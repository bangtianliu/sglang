"""Unselected tiled-weight trial: b128 operand-register loads, no weight LDS.

Packing uses the gfx1250 WMMA-v3 B lane/register mapping, k_width=8.
Input TDM copies retain four buffers; only immutable weights use .cs.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, cdna4, gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import chain_guard
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
)


@g.jit
def weight(W, tile, step, K: gl.constexpr, mma: gl.constexpr):
    layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [16, 0], [32, 0], [64, 0]],
        lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]],
        warp_bases=[[0, 0]],
        block_bases=[],
        shape=[128, 16],
    )
    k = gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
    n = gl.arange(0, 16, layout=gl.SliceLayout(0, layout))
    off = (
        tile * 16 * K
        + step * 2048
        + (k[:, None] // 16) * 256
        + (k[:, None] // 8 % 2 * 16 + n[None, :]) * 8
        + k[:, None] % 8
    )
    b = gl.load(W + off, cache_modifier=".cs")
    return gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))


@g.jit
def project(
    X,
    W,
    P,
    Status,
    Failure,
    N: gl.constexpr,
    K: gl.constexpr,
    STAGE: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Packed=None,
    SK: gl.constexpr = 4,
):
    tile = gl.program_id(0)
    delayed_point(Seed, tile, 0)
    allowed = chain_guard.enabled(Failure, STAGE)
    ok = allowed & (tile != SKIP)
    if ok:
        sh: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[128, 8]], [16, 128], [1, 0]
        )
        mma: gl.constexpr = AMDWMMALayout(
            version=3, transposed=True, warp_bases=[[0, 1]], instr_shape=[16, 16, 32]
        )
        sa = gl.allocate_shared_memory(gl.bfloat16, [4, 16, 128], sh)
        ad = gfx1250.tdm.make_tensor_descriptor(
            X, shape=(8, K), strides=(K, 1), block_shape=(16, 128), layout=sh
        )
        for step in gl.static_range(3):
            gfx1250.tdm.async_load(ad, [0, step * 128], sa.index(step))
        b0 = weight(W, tile, 0, K, mma)
        b1 = weight(W, tile, 1, K, mma)
        parts = ()
        acc = gl.full((16, 16), 0, gl.float32, mma)
        for step in gl.static_range(K // 128):
            if step + 3 < K // 128:
                gfx1250.tdm.async_load(
                    ad, [0, (step + 3) * 128], sa.index((step + 3) % 4)
                )
            if step + 2 < K // 128:
                b2 = weight(W, tile, step + 2, K, mma)
            gfx1250.tdm.async_wait(min(3, K // 128 - step - 1))
            aa = cdna4.async_copy.load_shared_relaxed(
                sa.index(step % 4), gl.DotOperandLayout(0, mma, 8)
            )
            if step % (K // SK // 128) == 0:
                acc = gl.full((16, 16), 0, gl.float32, mma)
            acc = gfx1250.wmma(aa, b0, acc)
            b0 = b1
            if step + 2 < K // 128:
                b1 = b2
            if (step + 1) % (K // SK // 128) == 0:
                parts += (acc,)
        gfx1250.tdm.async_wait(0)
        delayed_point(Seed, tile, 1)
        if SK == 8:
            value = ((parts[0] + parts[4]) + (parts[2] + parts[6])) + (
                (parts[1] + parts[5]) + (parts[3] + parts[7])
            )
        else:
            value = (parts[0] + parts[2]) + (parts[1] + parts[3])
        value = value.to(gl.bfloat16).to(gl.float32)
        r = gl.arange(0, 16, layout=gl.SliceLayout(1, mma))
        n = gl.arange(0, 16, layout=gl.SliceLayout(0, mma)) + tile * 16
        gl.store(
            P + r[:, None] * N + n[None, :],
            value.to(gl.int32, bitcast=True),
            r[:, None] < 8,
        )
        if Packed is not None:
            gl.store(
                Packed + r[:, None] * N + n[None, :],
                value.to(gl.bfloat16),
                r[:, None] < 8,
            )
    delayed_point(Seed, tile, 2)
    gl.store(Status + tile, ok)
    if allowed:
        chain_guard.finish_value(Failure, ok, tile, STAGE)

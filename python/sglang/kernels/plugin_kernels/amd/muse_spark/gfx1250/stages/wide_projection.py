"""Independent-launch BF16 projection: native-sized TDM ring, exact fixed-tree split-K math."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, cdna4, gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import chain_guard
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
)


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
    gl.static_assert(SK == 4 or SK == 8)
    gl.static_assert(K % (SK * 512) == 0)
    tile = gl.program_id(0)
    delayed_point(Seed, tile, 0)
    allowed = chain_guard.enabled(Failure, STAGE)
    ok = allowed & (tile != SKIP)
    if ok:
        # Matches AITER's native TN shared/operand configuration. Keep the four
        # independent K accumulations and fixed reduction tree of our reference.
        sh: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[512, 8]], [16, 512], [1, 0]
        )
        mma: gl.constexpr = AMDWMMALayout(
            version=3, transposed=True, warp_bases=[[0, 1]], instr_shape=[16, 16, 32]
        )
        sa = gl.allocate_shared_memory(gl.bfloat16, [4, 16, 512], sh)
        sb = gl.allocate_shared_memory(gl.bfloat16, [4, 16, 512], sh)
        ad = gfx1250.tdm.make_tensor_descriptor(
            X, shape=(8, K), strides=(K, 1), block_shape=(16, 512), layout=sh
        )
        bd = gfx1250.tdm.make_tensor_descriptor(
            W + tile * 16 * K,
            shape=(16, K),
            strides=(K, 1),
            block_shape=(16, 512),
            layout=sh,
        )
        for step in gl.static_range(3):
            gfx1250.tdm.async_load(ad, [0, step * 512], sa.index(step))
            gfx1250.tdm.async_load(bd, [0, step * 512], sb.index(step))
        parts = ()
        acc = gl.full((16, 16), 0, gl.float32, mma)
        for step in gl.static_range(K // 512):
            if step + 3 < K // 512:
                gfx1250.tdm.async_load(
                    ad, [0, (step + 3) * 512], sa.index((step + 3) % 4)
                )
                gfx1250.tdm.async_load(
                    bd, [0, (step + 3) * 512], sb.index((step + 3) % 4)
                )
            gfx1250.tdm.async_wait(min(3, K // 512 - step - 1) * 2)
            aa = cdna4.async_copy.load_shared_relaxed(
                sa.index(step % 4), gl.DotOperandLayout(0, mma, 8)
            )
            bb = cdna4.async_copy.load_shared_relaxed(
                sb.index(step % 4).permute([1, 0]), gl.DotOperandLayout(1, mma, 8)
            )
            if step % (K // SK // 512) == 0:
                acc = gl.full((16, 16), 0, gl.float32, mma)
            acc = gfx1250.wmma(aa, bb, acc)
            if (step + 1) % (K // SK // 512) == 0:
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

"""Native-inspired fused gate/up core with tiled, compiler-tracked activations.

Volatile intermediate loads lower to scope:SCOPE_SYS in the tested compiler,
which is stronger than the required device visibility but drains the pipeline.
A nonvolatile .cg experiment did not emit a verifiable device scope and was
rejected despite passing arithmetic checks. Call only after the existing private readiness flag succeeds.
No whole-row LDS cache; pure converter and compiler-tracked memory operations.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    experts_tracked as fp4,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import _mma


@g.jit
def input_tile(X, Sorted, block, step, BK: gl.constexpr, SECOND: gl.constexpr):
    mma: gl.constexpr = _mma(gl.num_warps(), 64)
    layout: gl.constexpr = gl.to_linear_layout(
        gl.DotOperandLayout(0, mma, 8), [4, 16, BK]
    )
    ss = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    mm = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    kk = gl.arange(0, BK, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    slot = gl.load(Sorted + block * 16 + mm)
    row = slot if SECOND else slot // 8
    ptr = (
        X
        + row[None, :, None] * 4096
        + ss[:, None, None] * 1024
        + step * BK
        + kk[None, None, :]
    )
    value = gl.load(
        ptr,
        slot[None, :, None] < 64,
        other=0,
        cache_modifier=".cg" if SECOND else "",
        volatile=SECOND,
    ).to(gl.bfloat16)
    return gl.convert_layout(value, gl.DotOperandLayout(0, mma, 8))


@g.jit
def gemm(
    X, Sorted, W, Scale, block, expert, tile, BK: gl.constexpr, SECOND: gl.constexpr
):
    mma: gl.constexpr = _mma(gl.num_warps(), 64)
    acc = gl.full((4, 16, 64), 0, gl.float32, mma)
    if not SECOND:
        acc_up = gl.full((4, 16, 64), 0, gl.float32, mma)
    a0 = input_tile(X, Sorted, block, 0, BK, SECOND)
    a1 = input_tile(X, Sorted, block, 1, BK, SECOND)
    p0, s0 = fp4.weight_packets(
        W, Scale, expert, tile, 0, 4096 if SECOND else 8192, 64, BK, 4
    )
    p1, s1 = fp4.weight_packets(
        W, Scale, expert, tile, 1, 4096 if SECOND else 8192, 64, BK, 4
    )
    p2, s2 = fp4.weight_packets(
        W, Scale, expert, tile, 2, 4096 if SECOND else 8192, 64, BK, 4
    )
    if not SECOND:
        u0, t0 = fp4.weight_packets(W, Scale, expert, tile + 64, 0, 8192, 64, BK, 4)
        u1, t1 = fp4.weight_packets(W, Scale, expert, tile + 64, 1, 8192, 64, BK, 4)
        u2, t2 = fp4.weight_packets(W, Scale, expert, tile + 64, 2, 8192, 64, BK, 4)
    for step in range(1024 // BK):
        b = fp4.upcast_native(p0, s0, 64, BK, 4)
        acc = gfx1250.wmma(a0, b, acc)
        if not SECOND:
            b_up = fp4.upcast_native(u0, t0, 64, BK, 4)
            acc_up = gfx1250.wmma(a0, b_up, acc_up)
            u0 = u1
            t0 = t1
            u1 = u2
            t1 = t2
            if step + 3 < 1024 // BK:
                u2, t2 = fp4.weight_packets(
                    W, Scale, expert, tile + 64, step + 3, 8192, 64, BK, 4
                )
        p0 = p1
        s0 = s1
        p1 = p2
        s1 = s2
        if step + 3 < 1024 // BK:
            p2, s2 = fp4.weight_packets(
                W, Scale, expert, tile, step + 3, 4096 if SECOND else 8192, 64, BK, 4
            )
        a0 = a1
        if step + 2 < 1024 // BK:
            a1 = input_tile(X, Sorted, block, step + 2, BK, SECOND)
    result = gl.sum(acc, 0)
    if not SECOND:
        up = gl.sum(acc_up, 0)
        result = result / (1.0 + gl.exp(-result)) * up
    return result

"""Compiler-tracked operand-aligned FP4 words for experimental Triton3.9 S5.

upcast uses pure ALU assembly. upcast_native borrows the compiler-supported
conversion for the larger fused core, whose fixed-register version hit an
LLVM allocator assertion. Neither function issues a memory instruction.
Integrated Triton3.8 tests failed; the public experimental wrapper requires3.9.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna5

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts_aligned import (
    byte_layout,
    word_layout,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import _mma


@g.jit
def weight_packets(
    W,
    Scale,
    expert,
    tile,
    step,
    ROWS: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    S: gl.constexpr,
):
    linear: gl.constexpr = gl.to_linear_layout(
        gl.DotOperandLayout(1, _mma(gl.num_warps(), BN), 8), [S, BK, BN]
    )
    layout: gl.constexpr = word_layout(linear)
    ss = gl.arange(0, S, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    kk = gl.arange(0, BK // 8, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    nn = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    k = ss[:, None, None] * (4096 // S) + step * BK + kk[None, :, None] * 8
    row = tile * BN + nn[None, None, :]
    ptr = W.to(gl.pointer_type(gl.uint32)) + expert * ROWS * 512 + row * 512 + k // 8
    raw = gl.load(ptr, cache_modifier=".cs")
    scale = gl.load(Scale + expert * ROWS * 128 + row * 128 + k // 32).to(gl.uint32)
    return raw, scale


@g.constexpr_function
def converter_asm():
    constraints = [f"=&{{v{24 + 4 * i + c}}}" for c in range(4) for i in range(4)] + [
        "v"
    ] * 8
    asm = ["s_wait_alu depctr_vm_vsrc(0)", "s_wait_alu depctr_va_vdst(0)"]
    asm += [
        f"v_cvt_scale_pk8_bf16_fp4 v[{24 + 4 * i}:{27 + 4 * i}], ${16 + i}, ${20 + i}"
        for i in range(4)
    ]
    asm += ["s_wait_alu depctr_va_vdst(0)"]
    return "\n".join(asm), ",".join(constraints)


@g.jit
def upcast(raw, scale, BN: gl.constexpr, BK: gl.constexpr, S: gl.constexpr):
    # Both lane halves share each 32-K scale in the WMMA B-operand mapping.
    payload = scale | (scale << 8)
    asm: gl.constexpr = converter_asm()
    a, b, c, d = gl.inline_asm_elementwise(
        asm[0], asm[1], [raw, payload], dtype=(gl.uint32,) * 4, is_pure=True, pack=4
    )
    pairs = (
        gl.join(gl.join(a, b), gl.join(c, d))
        .permute((0, 1, 4, 3, 2))
        .reshape((S, BK // 2, BN))
    )
    low = (pairs & 65535).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    high = (pairs >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    value = gl.join(low, high).permute((0, 1, 3, 2)).reshape((S, BK, BN))
    return gl.convert_layout(value, gl.DotOperandLayout(1, _mma(gl.num_warps(), BN), 8))


@g.jit
def upcast_native(raw, scale, BN: gl.constexpr, BK: gl.constexpr, S: gl.constexpr):
    linear: gl.constexpr = gl.to_linear_layout(
        gl.DotOperandLayout(1, _mma(gl.num_warps(), BN), 8), [S, BK, BN]
    )
    packed = (
        gl.join(
            gl.join(raw & 255, (raw >> 8) & 255), gl.join((raw >> 16) & 255, raw >> 24)
        )
        .permute((0, 1, 4, 3, 2))
        .reshape((S, BK // 2, BN))
        .to(gl.uint8)
    )
    packed = gl.convert_layout(packed, byte_layout(linear))
    repeat = gl.join(scale, scale)
    repeat = gl.join(repeat, repeat)
    repeat = gl.join(repeat, repeat)
    expanded = repeat.permute((0, 1, 3, 4, 5, 2)).reshape((S, BK, BN)).to(gl.uint8)
    expanded = gl.convert_layout(expanded, linear)
    value = cdna5.scaled_upcast(packed, expanded, gl.bfloat16, axis=1)
    return gl.convert_layout(value, gl.DotOperandLayout(1, _mma(gl.num_warps(), BN), 8))

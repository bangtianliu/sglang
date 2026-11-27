"""Experimental native FP4 loads directly into WMMA operand registers.

The operand has four noncontiguous packed words per lane. Batch those word
and scale reads before one LOADcnt/XCNT wait; this pins the memory layout
through inline assembly. Canonical byte unpacking and scale broadcasting
change register order only. No activation quantization or weight shuffling.

The explicit wait completes each packet batch before returning, so the three
staged packets in the caller do not overlap their memory latency with WMMA.
This candidate has zero weight LDS traffic but misses the performance target;
keep ALIGNED=False in the default stage until further review.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna5

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import _mma


@g.constexpr_function
def word_layout(linear):
    # Eight consecutive BF16 operand values become one packed FP4 word.
    assert linear.reg_bases[:3] == [[0, 1, 0], [0, 2, 0], [0, 4, 0]]

    def shrink(bases):
        return [[s, k // 8, n] for s, k, n in bases]

    return gl.DistributedLinearLayout(
        shrink(linear.reg_bases[3:]),
        shrink(linear.lane_bases),
        shrink(linear.warp_bases),
        shrink(linear.block_bases),
        [linear.shape[0], linear.shape[1] // 8, linear.shape[2]],
    )


@g.constexpr_function
def byte_layout(linear):
    def shrink(bases):
        return [[s, k // 2, n] for s, k, n in bases]

    return gl.DistributedLinearLayout(
        shrink(linear.reg_bases[1:]),
        shrink(linear.lane_bases),
        shrink(linear.warp_bases),
        shrink(linear.block_bases),
        [linear.shape[0], linear.shape[1] // 2, linear.shape[2]],
    )


@g.constexpr_function
def packet_asm():
    # Each lane owns four operand words. Issue all eight reads before one wait.
    asm = []
    for i in range(4):
        asm.append(f"global_load_b32 ${i}, ${8 + i}, off th:TH_LOAD_NT")
        asm.append(f"global_load_ubyte ${4 + i}, ${12 + i}, off")
    asm += ["s_wait_loadcnt 0", "s_wait_xcnt 0"]
    return "\n".join(asm), ",".join(["=&v"] * 8 + ["v"] * 8)


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
    gl.static_assert(BN == 64 and BK == 64 and S == 4 and gl.num_warps() == 16)
    mma: gl.constexpr = _mma(gl.num_warps(), BN)
    linear: gl.constexpr = gl.to_linear_layout(
        gl.DotOperandLayout(1, mma, 8), [S, BK, BN]
    )
    layout: gl.constexpr = word_layout(linear)
    ss = gl.arange(0, S, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    kk = gl.arange(0, BK // 8, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    nn = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    k = ss[:, None, None] * (4096 // S) + step * BK + kk[None, :, None] * 8
    row = tile * BN + nn[None, None, :]
    ptr = W.to(gl.pointer_type(gl.uint32)) + expert * ROWS * 512 + row * 512 + k // 8
    sptr = Scale + expert * ROWS * 128 + row * 128 + k // 32
    asm: gl.constexpr = packet_asm()
    raw, scale = gl.inline_asm_elementwise(
        asm[0],
        asm[1],
        [ptr.to(gl.uint64), sptr.to(gl.uint64)],
        dtype=(gl.uint32, gl.uint32),
        is_pure=False,
        pack=4,
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
    return packed, expanded


@g.jit
def upcast(packed, scale):
    value = cdna5.scaled_upcast(packed, scale, gl.bfloat16, axis=1)
    return gl.convert_layout(
        value, gl.DotOperandLayout(1, _mma(gl.num_warps(), packed.shape[2]), 8)
    )

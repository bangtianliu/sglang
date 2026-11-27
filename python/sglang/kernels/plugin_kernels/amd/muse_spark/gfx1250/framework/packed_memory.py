"""BF16 word packing on top of the audited agent-scope word instructions."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    vector_memory as vm,
)


@g.jit
def load_bf16_tile(Pointers, Mask, M: gl.constexpr, N: gl.constexpr):
    low_pointer, _ = gl.split(Pointers.reshape((M, N // 2, 2)))
    low_mask, _ = gl.split(Mask.broadcast_to((M, N)).reshape((M, N // 2, 2)))
    bits = load_tile_words(
        low_pointer.to(gl.pointer_type(gl.int32)), low_mask, M * N // 2
    ).reshape((M, N // 2))

    lo = (bits & 65535).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    hi = (bits >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    return gl.join(lo, hi).reshape((M, N))


@g.jit
def store_bf16_tile(Pointers, Value, Mask, M: gl.constexpr, N: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [1, 32], [gl.num_warps(), 1], [1, 0]
    )
    value = gl.convert_layout(Value.to(gl.bfloat16), layout)
    pointer = gl.convert_layout(Pointers, layout)
    mask = gl.convert_layout(Mask.broadcast_to((M, N)), layout)
    lo, hi = gl.split(value.reshape((M, N // 2, 2)))
    ptr, _ = gl.split(pointer.reshape((M, N // 2, 2)))
    valid, _ = gl.split(mask.reshape((M, N // 2, 2)))
    bits = lo.to(gl.uint16, bitcast=True).to(gl.uint32) | (
        hi.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )
    scoped.store_words(ptr.to(gl.pointer_type(gl.int32)), bits.to(gl.int32), valid)


@g.jit
def load_tile_words(Pointers, Mask, N: gl.constexpr):
    """Aligned contiguous quads, with uniform per-quad masks; one wait per batch."""
    layout: gl.constexpr = Pointers.type.layout
    ptr, _, _, _ = vm.split4(
        gl.convert_layout(
            Pointers.reshape((N,)), gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
        ),
        N // 4,
    )
    valid, _, _, _ = vm.split4(
        gl.convert_layout(
            Mask.broadcast_to(Pointers.shape).reshape((N,)),
            gl.BlockedLayout([4], [32], [gl.num_warps()], [0]),
        ),
        N // 4,
    )
    PACK: gl.constexpr = min(8, max(1, (N // 4) // (gl.num_warps() * 32)))
    asm: gl.constexpr = vm._load_asm(1, PACK)
    a, b, c, d, _ = gl.inline_asm_elementwise(
        asm[0],
        asm[1],
        [ptr, valid.to(gl.int32)],
        dtype=(gl.int32,) * 5,
        is_pure=False,
        pack=PACK,
    )
    return gl.convert_layout(
        vm.join4(a, b, c, d, N // 4).reshape(Pointers.shape), layout
    )


@g.jit
def store_tile_words(Pointers, Value, Mask, N: gl.constexpr):
    ptr, _, _, _ = vm.split4(
        gl.convert_layout(
            Pointers.reshape((N,)), gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
        ),
        N // 4,
    )
    valid, _, _, _ = vm.split4(
        gl.convert_layout(
            Mask.broadcast_to(Pointers.shape).reshape((N,)),
            gl.BlockedLayout([4], [32], [gl.num_warps()], [0]),
        ),
        N // 4,
    )
    a, b, c, d = vm.split4(
        gl.convert_layout(
            Value.reshape((N,)), gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
        ),
        N // 4,
    )
    PACK: gl.constexpr = min(8, max(1, (N // 4) // (gl.num_warps() * 32)))
    asm: gl.constexpr = vm._store_asm(1, PACK, True)
    gl.inline_asm_elementwise(
        asm[0],
        asm[1],
        [a, b, c, d, ptr, valid.to(gl.int32)],
        dtype=gl.int32,
        is_pure=False,
        pack=PACK,
    )


@g.jit
def store_bf16_vector(Base, Value, N: gl.constexpr):
    v = gl.convert_layout(
        Value.to(gl.bfloat16), gl.BlockedLayout([2], [32], [gl.num_warps()], [0])
    )
    lo, hi = gl.split(v.reshape((N // 2, 2)))
    bits = lo.to(gl.uint16, bitcast=True).to(gl.uint32) | (
        hi.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )
    j = gl.arange(0, N // 2, layout=bits.type.layout)
    store_tile_words(
        Base.to(gl.pointer_type(gl.int32)) + j, bits.to(gl.int32), j >= 0, N // 2
    )


@g.jit
def store_fp8_tile(Pointers, Value, Mask, N: gl.constexpr):
    l: gl.constexpr = gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
    v = gl.convert_layout(Value.to(gl.uint8, bitcast=True).reshape((N,)), l)
    p = gl.convert_layout(Pointers.reshape((N,)), l)
    m = gl.convert_layout(Mask.broadcast_to(Pointers.shape).reshape((N,)), l)
    a, b, c, d = vm.split4(v, N // 4)
    ptr, _, _, _ = vm.split4(p, N // 4)
    valid, _, _, _ = vm.split4(m, N // 4)
    bits = (
        a.to(gl.uint32)
        | (b.to(gl.uint32) << 8)
        | (c.to(gl.uint32) << 16)
        | (d.to(gl.uint32) << 24)
    )
    store_tile_words(
        ptr.to(gl.pointer_type(gl.int32)), bits.to(gl.int32), valid, N // 4
    )

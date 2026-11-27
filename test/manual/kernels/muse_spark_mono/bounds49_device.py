"""Generated once-only packed-boundary offset diagnostic; not a selected kernel."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    vector_memory as vm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    _join_row,
    _split_row,
)


@g.jit
def checked_ptr(
    Pointer, ID: gl.constexpr, Span: gl.constexpr, Desc, Records, Row, Mask
):
    scalar: gl.constexpr = len(Pointer.shape) == 0
    if scalar:
        address = gl.full(
            (1,), 0, gl.int64, gl.BlockedLayout([1], [32], [4], [0])
        ) + Pointer.to(gl.int64)
    else:
        address = Pointer.to(gl.int64)
    low = gl.load(Desc + ID * 2)
    high = gl.load(Desc + ID * 2 + 1)
    active = gl.full_like(address, True).to(gl.int1) & Mask
    valid = (address >= low) & (address + Span <= high)
    flat_bad = (active & ~valid).reshape((Pointer.numel,))
    count = gl.sum(flat_bad.to(gl.int32), 0)
    slot = (Row * 16 + ID) * 8
    gl.atomic_add(
        Records + slot + 7,
        gl.sum(active.reshape((Pointer.numel,)).to(gl.int32), 0).to(gl.int64),
        sem="relaxed",
        scope="gpu",
    )
    if count > 0:
        old = gl.atomic_cas(
            Records + slot, low * 0, low * 0 + 1, sem="relaxed", scope="gpu"
        )
        if old == 0:
            addr = address.reshape((Pointer.numel,))
            gl.store(
                Records + slot + 1,
                gl.min(gl.where(flat_bad, addr, 9223372036854775807), 0),
            )
            gl.store(Records + slot + 2, gl.max(gl.where(flat_bad, addr, 0), 0))
            gl.store(Records + slot + 3, low)
            gl.store(Records + slot + 4, high)
            gl.store(Records + slot + 5, Span)
            gl.store(Records + slot + 6, count)
    if scalar:
        return gl.sum(gl.where(valid, address, low), 0).to(Pointer.dtype)
    else:
        return gl.where(valid, address, low).to(Pointer.dtype)


@g.jit
def words_masked_checked(Pointer, Mask, ID: gl.constexpr, Desc, Records, Row):
    return scoped.load_words_masked(
        checked_ptr(Pointer, ID, 4, Desc, Records, Row, Mask), Mask
    )


@g.jit
def load_words_checked(
    Pointers, Mask, N: gl.constexpr, ID: gl.constexpr, Desc, Records, Row
):
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
    PACK: gl.constexpr = min(8, max(1, N // 4 // (gl.num_warps() * 32)))
    ptr = checked_ptr(ptr, ID, 16, Desc, Records, Row, valid)
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
def store_words_checked(
    Pointers, Value, Mask, N: gl.constexpr, ID: gl.constexpr, Desc, Records, Row
):
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
    PACK: gl.constexpr = min(8, max(1, N // 4 // (gl.num_warps() * 32)))
    ptr = checked_ptr(ptr, ID, 16, Desc, Records, Row, valid)
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
def store_bf16_checked(
    Base, Value, N: gl.constexpr, ID: gl.constexpr, Desc, Records, Row
):
    v = gl.convert_layout(
        Value.to(gl.bfloat16), gl.BlockedLayout([2], [32], [gl.num_warps()], [0])
    )
    lo, hi = gl.split(v.reshape((N // 2, 2)))
    bits = (
        lo.to(gl.uint16, bitcast=True).to(gl.uint32)
        | hi.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )
    j = gl.arange(0, N // 2, layout=bits.type.layout)
    store_words_checked(
        Base.to(gl.pointer_type(gl.int32)) + j,
        bits.to(gl.int32),
        j >= 0,
        N // 2,
        ID,
        Desc,
        Records,
        Row,
    )


@g.jit
def parameter_checked(Base, chunk, ID: gl.constexpr, Desc, Records, Row):
    block: gl.constexpr = 4
    i = chunk * 1024 + gl.arange(
        0, 1024, layout=gl.BlockedLayout([block], [32], [4], [0])
    )
    return gl.convert_layout(
        gl.load(
            checked_ptr(
                Base + i,
                ID,
                2 if Base.dtype.element_ty == gl.bfloat16 else 4,
                Desc,
                Records,
                Row,
                i >= 0,
            )
        ).to(gl.float32),
        gl.BlockedLayout([4], [32], [4], [0]),
    )


@g.jit
def residual_checked(Stream, Alpha, Beta, row, chunk, Desc, Records):
    q = gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [4], [0]))
    at = chunk * 1024 + q * 4
    asm: gl.constexpr = vm._load_asm(3, 2)
    words = gl.inline_asm_elementwise(
        asm[0],
        asm[1],
        [
            checked_ptr(
                Stream.to(gl.pointer_type(gl.int32)) + row * 8192 + at,
                1,
                16,
                Desc,
                Records,
                row,
                q >= 0,
            ),
            checked_ptr(
                Alpha.to(gl.pointer_type(gl.int32)) + at,
                2,
                16,
                Desc,
                Records,
                row,
                q >= 0,
            ),
            checked_ptr(
                Beta.to(gl.pointer_type(gl.int32)) + at,
                3,
                16,
                Desc,
                Records,
                row,
                q >= 0,
            ),
            (q >= 0).to(gl.int32),
        ],
        dtype=(gl.int32,) * 13,
        is_pure=False,
        pack=2,
    )
    stream = vm.join4(words[0], words[1], words[2], words[3], 256).to(
        gl.float32, bitcast=True
    )
    alpha = vm.join4(words[4], words[5], words[6], words[7], 256).to(
        gl.float32, bitcast=True
    )
    beta = vm.join4(words[8], words[9], words[10], words[11], 256).to(
        gl.float32, bitcast=True
    )
    return (stream, alpha, beta)


@g.jit
def packed_checked(
    row,
    Branch,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    Out,
    NewStream,
    POST_SCALE: gl.constexpr,
    PRE_BF16: gl.constexpr,
    WORDS: gl.constexpr,
    Trace=None,
    Value=None,
    Desc=None,
    Records=None,
):
    i = gl.arange(0, 8192, layout=gl.BlockedLayout([4], [32], [4], [0]))
    if Value is not None:
        branch = gl.convert_layout(Value, i.type.layout)
    elif WORDS:
        branch = load_words_checked(
            Branch + row * 8192 + i, i >= 0, 8192, 0, Desc, Records, row
        ).to(gl.float32, bitcast=True)
    else:
        branch = words_masked_checked(
            (Branch + row * 8192 + i).to(gl.pointer_type(gl.int32)),
            i >= 0,
            0,
            Desc,
            Records,
            row,
        ).to(gl.float32, bitcast=True)
    branch = branch.to(gl.bfloat16).to(gl.float32)
    if POST_SCALE:
        parts = _split_row(branch)
        scaled = ()
        for chunk in gl.static_range(8):
            scaled += (
                (parts[chunk] * parameter_checked(Post, chunk, 7, Desc, Records, row))
                .to(gl.bfloat16)
                .to(gl.float32),
            )
        branch = _join_row(scaled)
    total = gl.sum(
        gl.convert_layout(branch * branch, gl.BlockedLayout([1], [32], [4], [0])), 0
    )
    branch = (branch * gl.rsqrt(total / 8192 + 1e-08)).to(gl.bfloat16).to(gl.float32)
    parts = _split_row(branch)
    pieces = ()
    t_stream = gl.full((), 0, gl.uint64, gl.BlockedLayout([4], [32], [4], [0]))
    t_param, t_math, t_store = (t_stream, t_stream, t_stream)
    for chunk in gl.static_range(8):
        j = chunk * 1024 + gl.arange(
            0, 1024, layout=gl.BlockedLayout([4], [32], [4], [0])
        )
        if Alpha.dtype.element_ty == gl.float32 and Beta.dtype.element_ty == gl.float32:
            stream, alpha, beta = residual_checked(
                Stream, Alpha, Beta, row, chunk, Desc, Records
            )
        else:
            stream = load_words_checked(
                (Stream + row * 8192 + j).to(gl.pointer_type(gl.int32)),
                j >= 0,
                1024,
                1,
                Desc,
                Records,
                row,
            ).to(gl.float32, bitcast=True)
            alpha = parameter_checked(Alpha, chunk, 2, Desc, Records, row)
            beta = parameter_checked(Beta, chunk, 3, Desc, Records, row)
        updated = alpha * stream + beta * parts[chunk]
        pieces += (updated,)
    updated = _join_row(pieces)
    store_words_checked(
        NewStream.to(gl.pointer_type(gl.int32)) + row * 8192 + i,
        updated.to(gl.int32, bitcast=True),
        i >= 0,
        8192,
        6,
        Desc,
        Records,
        row,
    )
    pre = updated.to(gl.bfloat16).to(gl.float32) if PRE_BF16 else updated
    total = gl.sum(
        gl.convert_layout(pre * pre, gl.BlockedLayout([1], [32], [4], [0])), 0
    )
    inv = gl.rsqrt(total / 8192 + 1e-05)
    parts = _split_row(pre)
    t_gain = gl.full((), 0, gl.uint64, gl.BlockedLayout([4], [32], [4], [0]))
    t_apply, t_output = (t_gain, t_gain)
    outputs = ()
    for chunk in gl.static_range(8):
        if Pre is not None:
            gain = parameter_checked(Pre, chunk, 4, Desc, Records, row)
        else:
            gain = gl.full(
                (1024,), 1, gl.float32, gl.BlockedLayout([4], [32], [4], [0])
            )
        value = parts[chunk] * inv * gain
        outputs += (value,)
    value = _join_row(outputs)
    store_bf16_checked(Out + row * 8192, value, 8192, 5, Desc, Records, row)

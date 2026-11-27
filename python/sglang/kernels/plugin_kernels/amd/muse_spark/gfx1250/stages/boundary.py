"""Stage-local handoffs and Muse Spark's BF16 branch / FP32 stream boundary.

State belongs to serial launches on one stream. Every CTA has its own epoch
and private128-byte completion line per phase. Discard all outputs/state after
any false status. No allocation, model state, or peer mapping inside device code.
"""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
    sync,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    vector_memory as vm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_tile_words,
    store_bf16_vector,
    store_tile_words,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)


class StageState(msgspec.Struct):
    epochs: torch.Tensor
    arrivals: torch.Tensor
    flags: torch.Tensor
    status: torch.Tensor


def make_state(device, *, ctas, groups=1, phases=1):
    return StageState(
        torch.zeros((ctas, 32), device=device, dtype=torch.int32),
        torch.zeros((phases, groups, 32), device=device, dtype=torch.int32),
        torch.zeros((phases, ctas, 32), device=device, dtype=torch.int32),
        torch.empty(ctas, device=device, dtype=torch.int32),
    )


@g.jit
def handoff(Arrivals, Flags, tag, group, local, MEMBERS: gl.constexpr):
    """Drain all producer waves, count once, publish one exact tag per consumer."""
    scoped.drain_stores()
    ticket = _advance_tag(Arrivals, group)
    if ticket == tag * MEMBERS:
        scoped.publish_flags(Flags + group * MEMBERS * 32, tag, MEMBERS)
    seen, polls = scoped.wait_flag(
        Flags + (group * MEMBERS + local) * 32,
        tag,
        100_000_000,
        MAX_SPINS=4096,
        CLOCK_EVERY=64,
        SHADER_CLOCK=True,
    )
    return seen == tag


@g.jit
def store_projection(
    P, task, value, N: gl.constexpr, BN: gl.constexpr, ROUND: gl.constexpr
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 32], [gl.num_warps(), 1], [1, 0]
    )
    r = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    c = gl.arange(0, BN, layout=gl.SliceLayout(0, layout))
    value = gl.convert_layout(value, layout)
    if ROUND:
        value = value.to(gl.bfloat16).to(gl.float32)
    scoped.store_words(
        P + r[:, None] * N + task * BN + c[None, :],
        value.to(gl.int32, bitcast=True),
        r[:, None] < 8,
    )


@g.jit
def _split_four(values):
    matrix = gl.permute(values.reshape((1024, 2, 2)), (0, 2, 1))
    left, right = gl.split(matrix)
    a, b = gl.split(left)
    c, d = gl.split(right)
    layout: gl.constexpr = gl.BlockedLayout([4], [32], [4], [0])
    return (
        gl.convert_layout(a, layout),
        gl.convert_layout(b, layout),
        gl.convert_layout(c, layout),
        gl.convert_layout(d, layout),
    )


@g.jit
def _split_row(v):
    even, odd = gl.split(gl.permute(v.reshape((8, 1024)), (1, 0)).reshape((1024, 4, 2)))
    a0, a2, a4, a6 = _split_four(even.reshape((4096,)))
    a1, a3, a5, a7 = _split_four(odd.reshape((4096,)))
    return a0, a1, a2, a3, a4, a5, a6, a7


@g.jit
def _join_row(v):
    even = gl.join(gl.join(v[0], v[4]), gl.join(v[2], v[6])).reshape((1024, 4))
    odd = gl.join(gl.join(v[1], v[5]), gl.join(v[3], v[7])).reshape((1024, 4))
    return gl.convert_layout(
        gl.permute(gl.join(even, odd).reshape((1024, 8)), (1, 0)).reshape((8192,)),
        gl.BlockedLayout([4], [32], [4], [0]),
    )


@g.jit
def _parameter(Base, chunk):
    # Immutable parameter vectors: contiguous compiler-tracked vector loads.
    block: gl.constexpr = 4
    i = chunk * 1024 + gl.arange(
        0, 1024, layout=gl.BlockedLayout([block], [32], [4], [0])
    )
    return gl.convert_layout(
        gl.load(Base + i).to(gl.float32), gl.BlockedLayout([4], [32], [4], [0])
    )


@g.jit
def _residual_inputs(Stream, Alpha, Beta, row, chunk):
    # Reuse the audited b128 generator: three addresses, one wait for the batch.
    q = gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [4], [0]))
    at = chunk * 1024 + q * 4
    asm: gl.constexpr = vm._load_asm(3, 2)
    words = gl.inline_asm_elementwise(
        asm[0],
        asm[1],
        [
            Stream.to(gl.pointer_type(gl.int32)) + row * 8192 + at,
            Alpha.to(gl.pointer_type(gl.int32)) + at,
            Beta.to(gl.pointer_type(gl.int32)) + at,
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
    return stream, alpha, beta


@g.jit
def _norm_residual_scoped(
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
):
    if Trace is not None:
        gl.store(Trace, sync.clock())
    i = gl.arange(0, 8192, layout=gl.BlockedLayout([4], [32], [4], [0]))
    if Value is not None:
        branch = gl.convert_layout(Value, i.type.layout)
    elif WORDS:
        branch = load_tile_words(Branch + row * 8192 + i, i >= 0, 8192).to(
            gl.float32, bitcast=True
        )
    else:
        branch = scoped.load_words_masked(
            (Branch + row * 8192 + i).to(gl.pointer_type(gl.int32)), i >= 0
        ).to(gl.float32, bitcast=True)
    branch = branch.to(gl.bfloat16).to(gl.float32)
    if Trace is not None:
        gl.store(Trace + 3, sync.clock())
    if POST_SCALE:
        parts = _split_row(branch)
        scaled = ()
        for chunk in gl.static_range(8):
            scaled += (
                (parts[chunk] * _parameter(Post, chunk)).to(gl.bfloat16).to(gl.float32),
            )
        branch = _join_row(scaled)
    if Trace is not None:
        gl.store(Trace + 4, sync.clock())
    total = gl.sum(
        gl.convert_layout(branch * branch, gl.BlockedLayout([1], [32], [4], [0])), 0
    )
    if Trace is not None:
        gl.store(Trace + 10, sync.clock())
    branch = (branch * gl.rsqrt(total / 8192 + 1e-8)).to(gl.bfloat16).to(gl.float32)
    if Trace is not None:
        gl.store(Trace + 5, sync.clock())
    parts = _split_row(branch)
    pieces = ()
    t_stream = gl.full((), 0, gl.uint64, gl.BlockedLayout([4], [32], [4], [0]))
    t_param, t_math, t_store = t_stream, t_stream, t_stream
    for chunk in gl.static_range(8):
        j = chunk * 1024 + gl.arange(
            0, 1024, layout=gl.BlockedLayout([4], [32], [4], [0])
        )
        if Trace is not None:
            t0 = sync.clock()
        if Alpha.dtype.element_ty == gl.float32 and Beta.dtype.element_ty == gl.float32:
            stream, alpha, beta = _residual_inputs(Stream, Alpha, Beta, row, chunk)
            if Trace is not None:
                t1 = sync.clock()
        else:
            stream = load_tile_words(
                (Stream + row * 8192 + j).to(gl.pointer_type(gl.int32)), j >= 0, 1024
            ).to(gl.float32, bitcast=True)
            alpha = _parameter(Alpha, chunk)
            beta = _parameter(Beta, chunk)
            if Trace is not None:
                t1 = sync.clock()
        if Trace is not None:
            t2 = sync.clock()
        updated = alpha * stream + beta * parts[chunk]
        if Trace is not None:
            t3 = sync.clock()
        if Trace is not None:
            t4 = sync.clock()
            t_stream += t1 - t0
            t_param += t2 - t1
            t_math += t3 - t2
            t_store += t4 - t3
        pieces += (updated,)
    updated = _join_row(pieces)
    if Trace is not None:
        t0 = sync.clock()
    store_tile_words(
        NewStream.to(gl.pointer_type(gl.int32)) + row * 8192 + i,
        updated.to(gl.int32, bitcast=True),
        i >= 0,
        8192,
    )
    if Trace is not None:
        gl.store(Trace + 6, sync.clock())
        gl.store(Trace + 19, t_stream)
        gl.store(Trace + 20, t_param)
        gl.store(Trace + 21, t_math)
        gl.store(Trace + 22, sync.clock() - t0)
    pre = updated.to(gl.bfloat16).to(gl.float32) if PRE_BF16 else updated
    total = gl.sum(
        gl.convert_layout(pre * pre, gl.BlockedLayout([1], [32], [4], [0])), 0
    )
    if Trace is not None:
        gl.store(Trace + 13, sync.clock())
    inv = gl.rsqrt(total / 8192 + 1e-5)
    parts = _split_row(pre)
    t_gain = gl.full((), 0, gl.uint64, gl.BlockedLayout([4], [32], [4], [0]))
    t_apply, t_output = t_gain, t_gain
    outputs = ()
    for chunk in gl.static_range(8):
        if Trace is not None:
            t0 = sync.clock()
        if Pre is not None:
            gain = _parameter(Pre, chunk)
        else:
            gain = gl.full(
                (1024,), 1, gl.float32, gl.BlockedLayout([4], [32], [4], [0])
            )
        if Trace is not None:
            t1 = sync.clock()
        value = parts[chunk] * inv * gain
        if Trace is not None:
            t2 = sync.clock()
        outputs += (value,)
        if Trace is not None:
            t3 = sync.clock()
            t_gain += t1 - t0
            t_apply += t2 - t1
            t_output += t3 - t2
    value = _join_row(outputs)
    if Trace is not None:
        t0 = sync.clock()
    store_bf16_vector(Out + row * 8192, value, 8192)
    if Trace is not None:
        gl.store(Trace + 8, sync.clock())
        gl.store(Trace + 23, t_gain)
        gl.store(Trace + 24, t_apply)
        gl.store(Trace + 25, sync.clock() - t0)


@g.jit
def norm_residual(
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
    SCOPED_IO: gl.constexpr = False,
    PACKED_ROW: gl.constexpr = False,
):
    """Exact legacy default; packed experiment requires explicit opt-in.

    PACKED_ROW is known incorrect for the current S3 layout. Keep it disabled
    for S3; the isolated offset diagnostic did not validate packed numerics.
    """
    if PACKED_ROW and SCOPED_IO and gl.num_warps() == 4:
        _norm_residual_scoped(
            row,
            Branch,
            Stream,
            Post,
            Pre,
            Alpha,
            Beta,
            Out,
            NewStream,
            POST_SCALE,
            PRE_BF16,
            WORDS,
            Trace,
            Value,
        )
    else:
        i = gl.arange(
            0, 8192, layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
        )
        if Value is not None:
            branch = gl.convert_layout(Value, i.type.layout)
        elif WORDS:
            branch = load_tile_words(Branch + row * 8192 + i, i >= 0, 8192).to(
                gl.float32, bitcast=True
            )
        else:
            branch = scoped.load_words_masked(
                (Branch + row * 8192 + i).to(gl.pointer_type(gl.int32)), i >= 0
            ).to(gl.float32, bitcast=True)
        branch = branch.to(gl.bfloat16).to(gl.float32)
        if Trace is not None:
            gl.store(Trace + 3, sync.clock())
        if POST_SCALE:
            branch = (
                (branch * gl.load(Post + i).to(gl.float32))
                .to(gl.bfloat16)
                .to(gl.float32)
            )
        if Trace is not None:
            gl.store(Trace + 4, sync.clock())
        branch *= gl.rsqrt(gl.sum(branch * branch, 0) / 8192 + 1e-8)
        branch = branch.to(gl.bfloat16).to(gl.float32)
        if Trace is not None:
            gl.store(Trace + 5, sync.clock())
        stream = gl.load(
            Stream + row * 8192 + i, cache_modifier=".cg" if SCOPED_IO else ""
        )
        updated = gl.load(Alpha + i) * stream + gl.load(Beta + i) * branch
        if SCOPED_IO:
            store_tile_words(
                NewStream.to(gl.pointer_type(gl.int32)) + row * 8192 + i,
                updated.to(gl.int32, bitcast=True),
                i >= 0,
                8192,
            )
        else:
            gl.store(NewStream + row * 8192 + i, updated)
        if Trace is not None:
            gl.store(Trace + 6, sync.clock())
        pre = updated.to(gl.bfloat16).to(gl.float32) if PRE_BF16 else updated
        pre *= gl.rsqrt(gl.sum(pre * pre, 0) / 8192 + 1e-5)
        if Pre is not None:
            pre *= gl.load(Pre + i).to(gl.float32)
        if Trace is not None:
            gl.store(Trace + 7, sync.clock())
        if SCOPED_IO:
            store_bf16_vector(Out + row * 8192, pre, 8192)
        else:
            gl.store(Out + row * 8192 + i, pre.to(gl.bfloat16))
        if Trace is not None:
            gl.store(Trace + 8, sync.clock())


@g.jit
def norm_residual_packed(
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
    SCOPED_IO: gl.constexpr = False,
):
    norm_residual(
        row,
        Branch,
        Stream,
        Post,
        Pre,
        Alpha,
        Beta,
        Out,
        NewStream,
        POST_SCALE,
        PRE_BF16,
        WORDS,
        Trace,
        Value,
        SCOPED_IO,
        PACKED_ROW=True,
    )


@g.jit
def delayed_point(Seed, task, POINT: gl.constexpr):
    """Optional seeded per-CTA delay for handoff validation only."""
    if Seed is not None:
        seed = gl.load(Seed)
        count = ((task * 1103515245 + seed * (POINT + 1)) ^ (task >> 3)) & 31
        for _ in range(count):
            sync.backoff()


def validate_state(state, device, *, ctas, groups=1, phases=1):
    for tensor, shape in (
        (state.epochs, (ctas, 32)),
        (state.arrivals, (phases, groups, 32)),
        (state.flags, (phases, ctas, 32)),
        (state.status, (ctas,)),
    ):
        assert tensor.device == device and tensor.shape == shape
        assert tensor.dtype == torch.int32 and tensor.is_contiguous()

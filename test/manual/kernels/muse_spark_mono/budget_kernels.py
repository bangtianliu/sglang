"""3f measurement-only four-wave cores and LDS budget ballast; no runtime callers."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts_async import (
    expert_tile,
)


@g.constexpr_function
def pad_chunks(words):
    return tuple(1 << bit for bit in range(6, 16) if words & (1 << bit))


@g.jit
def pad_begin(Input, WORDS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    chunks: gl.constexpr = pad_chunks(WORDS)
    buffers = ()
    for chunk in gl.static_range(len(chunks)):
        buf = gl.allocate_shared_memory(
            gl.int32, [chunks[chunk]], gl.SwizzledSharedLayout(1, 1, 1, [0])
        )
        i = gl.arange(0, chunks[chunk], layout=layout)
        buf.store(gl.load(Input + i))
        buffers += (buf,)
    gl.barrier()
    return buffers


@g.jit
def pad_end(buffers, Sink):
    gl.barrier()
    value = gl.full((), 0, gl.int32, gl.BlockedLayout([1], [32], [4], [0]))
    for i in gl.static_range(len(buffers)):
        value += gl.sum(buffers[i].load(gl.BlockedLayout([1], [32], [4], [0])), 0)
    gl.store(Sink + gl.program_id(0), value)


@g.jit
def padding_only(Input, Sink, PAD_WORDS: gl.constexpr):
    buffers = pad_begin(Input, PAD_WORDS)
    pad_end(buffers, Sink)


@g.jit
def expert(
    X,
    Sorted,
    Experts,
    NumPost,
    W,
    Scale,
    Out,
    Pin,
    Sink,
    GRID: gl.constexpr,
    FIRST: gl.constexpr,
    PAD_WORDS: gl.constexpr,
):
    buffers = pad_begin(Pin, PAD_WORDS)
    total = gl.load(NumPost) // 16 * 64
    for task in range(gl.program_id(0), total, GRID):
        block = task // 64
        e = gl.load(Experts + block)
        if e >= 0:
            expert_tile(X, Sorted, W, Scale, Out, block, e, task % 64, FIRST)
    pad_end(buffers, Sink)


@g.jit
def skinny(
    X,
    W,
    Out,
    Pin,
    Sink,
    N: gl.constexpr,
    K: gl.constexpr,
    GRID: gl.constexpr,
    PAD_WORDS: gl.constexpr,
    SK: gl.constexpr = 16,
):
    buffers = pad_begin(Pin, PAD_WORDS)
    mma: gl.constexpr = AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        instr_shape=[16, 16, 32],
    )
    al: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [4, 8], [1, 4], [0, 1])
    ash: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[64, 8]], [16, 64], [1, 0]
    )
    bsh: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [64, 64], [0, 1]
    )
    sa = gl.allocate_shared_memory(gl.bfloat16, [2, 16, 64], ash)
    sb = gl.allocate_shared_memory(gl.bfloat16, [2, 64, 64], bsh)
    row = gl.arange(0, 16, layout=gl.SliceLayout(1, al))
    ak = gl.arange(0, 64, layout=gl.SliceLayout(0, al))
    bk = gl.arange(0, 64, layout=gl.SliceLayout(1, bl))
    bn = gl.arange(0, 64, layout=gl.SliceLayout(0, bl))
    for task in range(gl.program_id(0), (N // 64) * SK, GRID):
        split = task % SK
        tile = task // SK
        start = split * (K // SK)
        ap = row[:, None] * K + start + ak[None, :]
        bp = (tile * 64 + bn[None, :]) * K + start + bk[:, None]
        amask = (row[:, None] < 8).broadcast_to((16, 64))
        gfx1250.async_copy.global_to_shared(sa.index(0), X + ap, amask, 0)
        gfx1250.async_copy.global_to_shared(sb.index(0), W + bp)
        gfx1250.async_copy.commit_group()
        acc = gl.full((16, 64), 0, gl.float32, mma)
        for step in range(K // SK // 64):
            bank = step % 2
            gfx1250.async_copy.wait_group(0)
            gl.barrier()
            if step + 1 < K // SK // 64:
                gfx1250.async_copy.global_to_shared(
                    sa.index(1 - bank), X + ap + (step + 1) * 64, amask, 0
                )
                gfx1250.async_copy.global_to_shared(
                    sb.index(1 - bank), W + bp + (step + 1) * 64
                )
                gfx1250.async_copy.commit_group()
            a = sa.index(bank).load(gl.DotOperandLayout(0, mma, 8))
            b = sb.index(bank).load(gl.DotOperandLayout(1, mma, 8))
            acc = gfx1250.wmma(a, b, acc)
            gl.barrier()
        gfx1250.async_copy.wait_group(0)
        r = gl.arange(0, 16, layout=gl.SliceLayout(1, mma))
        n = gl.arange(0, 64, layout=gl.SliceLayout(0, mma)) + tile * 64
        gl.store(Out + split * 8 * N + r[:, None] * N + n[None, :], acc, r[:, None] < 8)
    pad_end(buffers, Sink)


@g.jit
def attention(
    Q,
    K,
    V,
    Pages,
    Length,
    KS,
    VS,
    Out,
    LSE,
    Pin,
    Sink,
    GRID: gl.constexpr,
    PAD_WORDS: gl.constexpr,
    SPLITS: gl.constexpr,
    LEFT: gl.constexpr,
    QUERY_ROWS: gl.constexpr = 16,
):
    buffers = pad_begin(Pin, PAD_WORDS)
    mma: gl.constexpr = AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]] if QUERY_ROWS == 16 else [[1, 0], [2, 0]],
        instr_shape=[16, 16, 32],
    )
    ql: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    kl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    qm = gl.arange(0, QUERY_ROWS, layout=gl.SliceLayout(1, ql))
    qd = gl.arange(0, 64, layout=gl.SliceLayout(0, ql))
    kt = gl.arange(0, 64, layout=gl.SliceLayout(1, kl))
    kd = gl.arange(0, 64, layout=gl.SliceLayout(0, kl))
    row = gl.arange(0, QUERY_ROWS, layout=gl.SliceLayout(1, mma))
    col = gl.arange(0, 64, layout=gl.SliceLayout(0, mma))
    length = gl.load(Length)
    first = gl.maximum(0, length - 8 - LEFT) // 64 * 64 if LEFT >= 0 else 0
    tiles = gl.cdiv(length - first, 64)
    scale = 0.125 * 1.4426950408889634 * gl.load(KS)
    GROUPS: gl.constexpr = 64 // QUERY_ROWS
    for task in range(gl.program_id(0), SPLITS * 4 * GROUPS, GRID):
        split = task // (4 * GROUPS)
        head = task % (4 * GROUPS) // GROUPS
        group = task % GROUPS
        qr = group * QUERY_ROWS + qm
        q = gl.load(
            Q
            + (qr[:, None] // 8) * 2048
            + (head * 8 + qr[:, None] % 8) * 64
            + qd[None, :]
        )
        q = gl.convert_layout(q, gl.DotOperandLayout(0, mma, 8))
        acc = gl.full((QUERY_ROWS, 64), 0, gl.float32, mma)
        maximum = gl.full(
            (QUERY_ROWS,), float("-inf"), gl.float32, gl.SliceLayout(1, mma)
        )
        denom = gl.full((QUERY_ROWS,), 0, gl.float32, gl.SliceLayout(1, mma))
        start = first + split * tiles // SPLITS * 64
        end = gl.minimum(length, first + (split + 1) * tiles // SPLITS * 64)
        for offset in range(start, end, 64):
            pos = offset + kt
            page = gl.load(Pages + pos // 16)
            ptr = (page * 16 + pos % 16)[:, None] * 256 + head * 64 + kd[None, :]
            k = gl.load(K + ptr).to(gl.bfloat16)
            v = gl.load(V + ptr).to(gl.bfloat16)
            score = (
                gfx1250.wmma(
                    q,
                    gl.convert_layout(
                        k.permute((1, 0)), gl.DotOperandLayout(1, mma, 8)
                    ),
                    gl.full((QUERY_ROWS, 64), 0, gl.float32, mma),
                )
                * scale
            )
            absolute = length - 8 + (group * QUERY_ROWS + row) // 8
            mask = (offset + col[None, :] <= absolute[:, None]) & (
                offset + col[None, :] < length
            )
            if LEFT >= 0:
                mask &= offset + col[None, :] >= absolute[:, None] - LEFT
            score = gl.where(mask, score, float("-inf"))
            new_max = gl.maximum(maximum, gl.max(score, 1))
            safe = gl.where(new_max == float("-inf"), 0.0, new_max)
            probability = gl.exp2(score - safe[:, None])
            alpha = gl.exp2(maximum - safe)
            denom = denom * alpha + gl.sum(probability, 1)
            acc *= alpha[:, None]
            acc = gfx1250.wmma(
                gl.convert_layout(
                    probability.to(gl.bfloat16), gl.DotOperandLayout(0, mma, 8)
                ),
                gl.convert_layout(v, gl.DotOperandLayout(1, mma, 8)),
                acc,
            )
            maximum = new_max
        value = acc * (gl.load(VS) / gl.where(denom > 0, denom, 1.0))[:, None]
        logs = gl.where(denom > 0, maximum + gl.log2(denom), float("-inf"))
        gl.store(Out + task * QUERY_ROWS * 64 + row[:, None] * 64 + col[None, :], value)
        gl.store(LSE + task * QUERY_ROWS + row, logs)
    pad_end(buffers, Sink)

"""Four-wave BF16 projection tiles with explicit, fixed K-split reduction order."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    vector_memory as vm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    store_bf16_tile,
    store_tile_words,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.timeline import (
    stamp,
)


@g.jit
def _copy_a(
    X, ap, step, bank, sa, mask, BK: gl.constexpr, SCOPED: gl.constexpr, Gate=None
):
    if Gate is not None:
        x = gl.load(
            X + ap + step * BK, mask, 0, cache_modifier=".cg" if SCOPED else ""
        ).to(gl.float32)
        gate = gl.load(
            Gate + ap + step * BK, mask, 0, cache_modifier=".cg" if SCOPED else ""
        ).to(gl.float32)
        xx = x.reshape((16, BK // 64, 64))
        inv = gl.rsqrt(gl.sum(xx * xx, 2) / 64 + 1e-5)
        value = (xx * inv[:, :, None]).reshape((16, BK)) / (1.0 + gl.exp(-gate))
        sa.index(bank).store(value.to(gl.bfloat16))
    elif SCOPED:
        gfx1250.async_copy.global_to_shared(
            sa.index(bank), X + ap + step * BK, mask, 0, cache_modifier=".cg"
        )
    else:
        gfx1250.async_copy.global_to_shared(sa.index(bank), X + ap + step * BK, mask, 0)


@g.jit
def partial_tile(
    X,
    W,
    Partials,
    task,
    N: gl.constexpr,
    K: gl.constexpr,
    SK: gl.constexpr,
    BK: gl.constexpr,
    SCOPED_INPUT: gl.constexpr = False,
    Trace=None,
    Gate=None,
):
    mma: gl.constexpr = AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        instr_shape=[16, 16, 32],
    )
    al: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [4, 8], [1, 4], [0, 1])
    ash: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BK, 8]], [16, BK], [1, 0]
    )
    bsh: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [BK, 64], [0, 1]
    )
    sa = gl.allocate_shared_memory(gl.bfloat16, [2, 16, BK], ash)
    sb = gl.allocate_shared_memory(gl.bfloat16, [2, BK, 64], bsh)
    row = gl.arange(0, 16, layout=gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, layout=gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, layout=gl.SliceLayout(1, bl))
    bn = gl.arange(0, 64, layout=gl.SliceLayout(0, bl))
    split = task % SK
    tile = task // SK
    ap = row[:, None] * K + split * (K // SK) + ak[None, :]
    bp = (tile * 64 + bn[None, :]) * K + split * (K // SK) + bk[:, None]
    mask = (row[:, None] < 8).broadcast_to((16, BK))
    stamp(Trace, gl.program_id(0), 24)
    _copy_a(X, ap, 0, 0, sa, mask, BK, SCOPED_INPUT, Gate)
    gfx1250.async_copy.global_to_shared(sb.index(0), W + bp)
    gfx1250.async_copy.commit_group()
    stamp(Trace, gl.program_id(0), 25)
    acc = gl.full((16, 64), 0, gl.float32, mma)
    for step in range(K // SK // BK):
        bank = step % 2
        gfx1250.async_copy.wait_group(0)
        gl.barrier()
        if step + 1 < K // SK // BK:
            _copy_a(X, ap, step + 1, 1 - bank, sa, mask, BK, SCOPED_INPUT, Gate)
            gfx1250.async_copy.global_to_shared(
                sb.index(1 - bank), W + bp + (step + 1) * BK
            )
            gfx1250.async_copy.commit_group()
        acc = gfx1250.wmma(
            sa.index(bank).load(gl.DotOperandLayout(0, mma, 8)),
            sb.index(bank).load(gl.DotOperandLayout(1, mma, 8)),
            acc,
        )
        gl.barrier()
    gfx1250.async_copy.wait_group(0)
    stamp(Trace, gl.program_id(0), 26)
    r = gl.arange(0, 16, layout=gl.SliceLayout(1, mma))
    n = gl.arange(0, 64, layout=gl.SliceLayout(0, mma)) + tile * 64
    store_tile_words(
        Partials + split * 8 * N + r[:, None] * N + n[None, :],
        acc.to(gl.int32, bitcast=True),
        r[:, None] < 8,
        16 * 64,
    )

    stamp(Trace, gl.program_id(0), 27)


@g.jit
def reduce_tile(
    Partials, Projection, tile, N: gl.constexpr, SK: gl.constexpr, ROUND: gl.constexpr
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    row = gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
    col = gl.arange(0, 64, layout=gl.SliceLayout(0, layout)) + tile * 64
    q = gl.arange(0, 128, layout=gl.BlockedLayout([1], [32], [4], [0]))
    words = vm.load_sources(
        Partials + (q // 16) * N + tile * 64 + (q % 16) * 4, q >= 0, 8 * N, SK, 1
    )
    values = ()
    for s in gl.static_range(SK):
        value = vm.join4(
            words[s * 4], words[s * 4 + 1], words[s * 4 + 2], words[s * 4 + 3], 128
        ).reshape((8, 64))
        values += (gl.convert_layout(value, layout).to(gl.float32, bitcast=True),)
    # Existing skinny core reduces split bits from high to low (verified by probe).
    if SK == 8:
        value = ((values[0] + values[4]) + (values[2] + values[6])) + (
            (values[1] + values[5]) + (values[3] + values[7])
        )
    else:
        gl.static_assert(SK == 4)
        value = (values[0] + values[2]) + (values[1] + values[3])
    if ROUND:
        value = value.to(gl.bfloat16).to(gl.float32)
    store_tile_words(
        Projection + row[:, None] * N + col[None, :],
        value.to(gl.int32, bitcast=True),
        row[:, None] >= 0,
        8 * 64,
    )
    return value


@g.jit
def project(
    pid,
    X,
    W,
    Partials,
    Projection,
    Barriers,
    tag,
    N: gl.constexpr,
    K: gl.constexpr,
    SK: gl.constexpr,
    BK: gl.constexpr,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    ROUND: gl.constexpr,
    SCOPED_INPUT: gl.constexpr = False,
    SKIP: gl.constexpr = -1,
    Packed=None,
    Trace=None,
):
    stamp(Trace, pid, 8)
    if pid != SKIP:
        for task in range(pid, (N // 64) * SK, GRID):
            partial_tile(X, W, Partials, task, N, K, SK, BK, SCOPED_INPUT, Trace)
    stamp(Trace, pid, 9)
    ok = gl.full((), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    if pid != SKIP:
        ok, _, _ = scoped.counter_barrier(
            Barriers, tag, 10_000_000, GRID, 12, MAX_SPINS=1_048_576
        )
    stamp(Trace, pid, 10)
    if ok:
        for tile in range(pid, N // 64, GRID):
            value = reduce_tile(Partials, Projection, tile, N, SK, ROUND)
            if Packed is not None:
                layout: gl.constexpr = value.type.layout
                r = gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
                c = gl.arange(0, 64, layout=gl.SliceLayout(0, layout)) + tile * 64
                store_bf16_tile(
                    Packed + r[:, None] * N + c[None, :], value, r[:, None] >= 0, 8, 64
                )
        stamp(Trace, pid, 11)
        ok, _, _ = scoped.counter_barrier(
            Barriers + STRIDE, tag, 10_000_000, GRID, 12, MAX_SPINS=1_048_576
        )
    stamp(Trace, pid, 12)
    return ok


@g.jit
def project_last(
    pid,
    X,
    W,
    Partials,
    Projection,
    Counters,
    Flags,
    tag,
    N: gl.constexpr,
    K: gl.constexpr,
    SK: gl.constexpr,
    BK: gl.constexpr,
    GRID: gl.constexpr,
    ROUND: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Packed=None,
    Trace=None,
    Gate=None,
):
    """Last split merges in fixed order; last N tile releases eight private row owners."""
    stamp(Trace, pid, 8)
    if pid != SKIP:
        for task in range(pid, (N // 64) * SK, GRID):
            tile = task // SK
            partial_tile(X, W, Partials, task, N, K, SK, BK, False, Trace, Gate)
            stamp(Trace, pid, 9)
            scoped.drain_stores()
            ticket = gl.atomic_add(Counters + tile * 32, 1, sem="relaxed", scope="gpu")
            stamp(Trace, pid, 10)
            if ticket + 1 == tag * SK:
                value = reduce_tile(Partials, Projection, tile, N, SK, ROUND)
                if Packed is not None:
                    l: gl.constexpr = value.type.layout
                    r = gl.arange(0, 8, layout=gl.SliceLayout(1, l))
                    c = gl.arange(0, 64, layout=gl.SliceLayout(0, l)) + tile * 64
                    store_bf16_tile(
                        Packed + r[:, None] * N + c[None, :],
                        value,
                        r[:, None] >= 0,
                        8,
                        64,
                    )
                stamp(Trace, pid, 11)
                scoped.drain_stores()
                merged = gl.atomic_add(
                    Counters + (N // 64) * 32, 1, sem="relaxed", scope="gpu"
                )
                if merged + 1 == tag * (N // 64):
                    scoped.publish_flags(Flags, tag, 8)
    ok = pid != SKIP
    if pid < 8:
        seen, _ = scoped.wait_flag(
            Flags + pid * 32, tag, 10_000_000, MAX_SPINS=1_048_576
        )
        ok = ok & (seen == tag)
    stamp(Trace, pid, 12)
    return ok

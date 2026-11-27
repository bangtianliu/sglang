"""M8 native BF16 GEMM device core; K partials reduce inside each CTA."""

import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, gfx1250


@g.constexpr_function
def _mma(waves, bn):
    splits = waves // (bn // 16)
    bases = [[1 << i, 0, 0] for i in range(splits.bit_length() - 1)]
    bases += [[0, 0, 1 << i] for i in range((bn // 16).bit_length() - 1)]
    return AMDWMMALayout(
        version=3, transposed=True, warp_bases=bases, instr_shape=[16, 16, 32], rank=3
    )


@g.jit
def load_input(X, K: gl.constexpr, S: gl.constexpr, BK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 32], [S, gl.num_warps() // S], [1, 0]
    )
    ss = gl.arange(0, S, layout=gl.SliceLayout(1, layout))
    j = gl.arange(0, 8 * K // S, layout=gl.SliceLayout(0, layout))
    x = gl.load(
        X
        + (j[None, :] // (K // S)) * K
        + ss[:, None] * (K // S)
        + j[None, :] % (K // S)
    )
    shared = gl.allocate_shared_memory(
        gl.bfloat16, [S, 8 * K // S], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    shared.store(x)
    return shared


@g.jit
def _input(
    cached_x,
    step,
    K: gl.constexpr,
    BK: gl.constexpr,
    S: gl.constexpr,
    mma: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 32], [S, gl.num_warps() // S], [1, 0]
    )
    j = gl.arange(0, 16 * BK, layout=gl.SliceLayout(0, layout))
    index = (
        gl.full((S, 16 * BK), 0, gl.int32, layout)
        + ((j[None, :] // BK) % 8) * (K // S)
        + step * BK
        + j[None, :] % BK
    )
    a = cached_x.gather(index, 1)
    a = gl.where(j[None, :] // BK < 8, a, 0)
    a = gl.reshape(a, (S, 16, BK))
    return gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))


@g.jit
def _weight(
    W,
    block,
    step,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    S: gl.constexpr,
    VEC: gl.constexpr,
    CACHE: gl.constexpr,
    DIRECT: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, VEC, 1], [1, 8, 4], [S, 1, gl.num_warps() // S], [1, 2, 0]
    )
    ss = gl.arange(0, S, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    kk = gl.arange(0, BK, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    nn = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    pointer = (
        W
        + (block * BN + nn[None, None, :]) * K
        + ss[:, None, None] * (K // S)
        + step * BK
        + kk[None, :, None]
    )
    if DIRECT:
        pointer = gl.convert_layout(
            pointer, gl.DotOperandLayout(1, _mma(gl.num_warps(), BN), 8)
        )
    return gl.load(pointer, cache_modifier=CACHE)


@g.jit
def skinny_tile(
    W,
    cached_x,
    block,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    S: gl.constexpr,
    VEC: gl.constexpr,
    PREFETCH: gl.constexpr,
    CACHE: gl.constexpr,
    DIRECT: gl.constexpr,
):
    mma: gl.constexpr = _mma(gl.num_warps(), BN)
    acc = gl.full((S, 16, BN), 0, gl.float32, mma)
    if PREFETCH >= 2:
        b0 = _weight(W, block, 0, N, K, BN, BK, S, VEC, CACHE, DIRECT)
        b1 = _weight(W, block, 1, N, K, BN, BK, S, VEC, CACHE, DIRECT)
        if PREFETCH >= 3:
            b2 = _weight(W, block, 2, N, K, BN, BK, S, VEC, CACHE, DIRECT)
        for step in (gl.static_range if K <= 4096 else range)(K // S // BK):
            a = _input(cached_x, step, K, BK, S, mma)
            b = gl.convert_layout(b0, gl.DotOperandLayout(1, mma, 8))
            acc = gfx1250.wmma(a, b, acc)
            b0 = b1
            if PREFETCH >= 3:
                b1 = b2
                if step + 3 < K // S // BK:
                    b2 = _weight(
                        W, block, step + 3, N, K, BN, BK, S, VEC, CACHE, DIRECT
                    )
            elif step + 2 < K // S // BK:
                b1 = _weight(W, block, step + 2, N, K, BN, BK, S, VEC, CACHE, DIRECT)
    else:
        for step in (gl.static_range if K <= 4096 else range)(K // S // BK):
            a = _input(cached_x, step, K, BK, S, mma)
            b = _weight(W, block, step, N, K, BN, BK, S, VEC, CACHE, DIRECT)
            b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
            acc = gfx1250.wmma(a, b, acc)
    # Fixed balanced tree in ascending wave/K-split order.
    return gl.sum(acc, 0)


@g.jit
def skinny_stage(
    task_index,
    ctas,
    X,
    W,
    Out,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    VEC: gl.constexpr,
    PREFETCH: gl.constexpr,
    CACHE: gl.constexpr,
    DIRECT: gl.constexpr,
):
    if task_index < N // BN:
        S: gl.constexpr = gl.num_warps() // (BN // 16)
        cached_x = load_input(X, K, S, BK)
        for task in range(task_index, N // BN, ctas):
            value = skinny_tile(
                W, cached_x, task, N, K, BN, BK, S, VEC, PREFETCH, CACHE, DIRECT
            )
            layout: gl.constexpr = gl.SliceLayout(0, _mma(gl.num_warps(), BN))
            row = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
            col = gl.arange(0, BN, layout=gl.SliceLayout(0, layout))
            gl.store(
                Out + row[:, None] * N + task * BN + col[None, :],
                value,
                row[:, None] < 8,
            )


@g.jit
def _standalone(
    X,
    W,
    Out,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    VEC: gl.constexpr,
    PREFETCH: gl.constexpr,
    CACHE: gl.constexpr,
    DIRECT: gl.constexpr,
):
    skinny_stage(
        gl.program_id(0),
        gl.num_programs(0),
        X,
        W,
        Out,
        N,
        K,
        BN,
        BK,
        VEC,
        PREFETCH,
        CACHE,
        DIRECT,
    )


def skinny_gemm(
    x,
    weight,
    *,
    waves=16,
    block_n=None,
    block_k=None,
    vector=8,
    prefetch=2,
    cache=".cs",
    direct=False,
    ctas=192,
):
    n, k = weight.shape
    block_n = (32 if n == 4608 else 64) if block_n is None else block_n
    block_k = (64 if n == 4608 else 128) if block_k is None else block_k
    assert x.shape == (8, k) and (n, k) in {(4608, 8192), (8192, 2048), (8192, 4096)}
    assert (
        x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
    )
    assert x.device == weight.device and waves in (4, 8, 16)
    assert block_n in (16, 32, 64) and waves >= block_n // 16
    assert vector in (4, 8) and prefetch in (1, 2, 3)
    assert not (waves == 8 and block_n == 16 and block_k == 256)
    assert (
        n % block_n == 0
        and k % ((waves // (block_n // 16)) * block_k) == 0
        and k // (waves // (block_n // 16)) // block_k >= max(1, prefetch)
    )
    assert 0 < ctas <= torch.cuda.get_device_properties(x.device).multi_processor_count
    out = torch.empty((8, n), device=x.device, dtype=torch.bfloat16)
    _standalone[(ctas,)](
        x,
        weight,
        out,
        n,
        k,
        block_n,
        block_k,
        vector,
        prefetch,
        cache,
        direct,
        num_warps=waves,
    )
    return out

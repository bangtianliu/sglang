"""TP4 S2: all-head context tasks, prefetched FP8 pages and one-launch merge.

Eight BF16 query rows, Q32/KV4, D64, E4M3 NHD page16; causal full or left2047.
Workspace is serial-stream owned. Check all256 status words and discard output
and workspace after timeout. Each merge consumer owns private128-byte flag lines.
Page-aligned SWA mapping is an explicit caller guarantee; arbitrary token maps
use the fallback. No peer memory or whole-cache fences.
"""

import msgspec
import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
    sync,
    vector_memory,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    store_bf16_vector,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)


class AttentionWorkspace(msgspec.Struct):
    partial: torch.Tensor
    lse: torch.Tensor
    counters: torch.Tensor
    flags: torch.Tensor
    status: torch.Tensor
    splits: int
    ctas: int
    pages_per_chunk: int


def make_workspace(
    device, *, context, window_left=-1, ctas=192, block_n=64, sliding_splits=8
):
    assert 8 <= context <= 131072 and window_left in (-1, 2047) and 64 <= ctas <= 192
    assert block_n in (32, 64, 128)
    active = context if window_left < 0 else min(context, window_left + 8)
    tiles = triton.cdiv(active, block_n)
    upper_tiles = triton.cdiv(
        active + (block_n - 1 if window_left >= 0 else 0), block_n
    )
    assert sliding_splits in (0, 4, 8)
    splits = (
        sliding_splits
        if window_left >= 0 and sliding_splits
        else min(ctas, max(16, tiles))
    )
    pages = triton.next_power_of_2(triton.cdiv(upper_tiles, splits) * block_n // 16)
    return AttentionWorkspace(
        torch.empty((splits, 256, 64), device=device, dtype=torch.int32),
        torch.empty((splits, 256), device=device, dtype=torch.int32),
        torch.zeros(
            (ctas + triton.cdiv(ctas, 8) + 1, 32), device=device, dtype=torch.int32
        ),
        torch.zeros((256, 32), device=device, dtype=torch.int32),
        torch.empty(256, device=device, dtype=torch.int32),
        splits,
        ctas,
        pages,
    )


@g.constexpr_function
def _vector_asm(pack, store):
    if store:
        constraints = ["=&s"] * pack
        constraints += [f"{{v{4 + p * 4 + c}}}" for c in range(4) for p in range(pack)]
        constraints += ["v"] * pack
        asm = [f"s_mov_b32 ${p}, 0" for p in range(pack)]
        asm += [
            f"global_store_b128 ${pack * 5 + p}, v[{4 + p * 4}:{7 + p * 4}], off scope:SCOPE_DEV"
            for p in range(pack)
        ]
        asm += ["s_wait_storecnt 0", "s_wait_xcnt 0x0"]
    else:
        constraints = [
            f"=&{{v{4 + p * 4 + c}}}" for c in range(4) for p in range(pack)
        ] + ["v"] * pack
        asm = [
            f"global_load_b128 v[{4 + p * 4}:{7 + p * 4}], ${pack * 4 + p}, off scope:SCOPE_DEV"
            for p in range(pack)
        ]
        asm += ["s_wait_loadcnt 0", "s_wait_xcnt 0x0"]
    return "\n".join(asm), ",".join(constraints)


@triton.constexpr_function
def _merge_power_of_two(value):
    return 1 << (value - 1).bit_length()


@g.jit
def _store128(ptr, a, b, c, d, PACKETS: gl.constexpr = 4096):
    PACK: gl.constexpr = min(16, PACKETS // (gl.num_warps() * 32))
    asm: gl.constexpr = _vector_asm(PACK, True)
    return gl.inline_asm_elementwise(
        asm[0], asm[1], [a, b, c, d, ptr], dtype=gl.int32, is_pure=False, pack=PACK
    )


@g.jit
def _load128(ptr, PACK: gl.constexpr):
    asm: gl.constexpr = _vector_asm(PACK, False)
    return gl.inline_asm_elementwise(
        asm[0], asm[1], [ptr], dtype=(gl.int32,) * 4, is_pure=False, pack=PACK
    )


@g.jit
def _cached_pages(
    PAGES, MAP, first, end, MAX_PAGES: gl.constexpr, MAP_PAGE: gl.constexpr
):
    layout: gl.constexpr = gl.SliceLayout(
        1, gl.BlockedLayout([1, 16], [2, 16], [gl.num_warps(), 1], [1, 0])
    )
    ids = gl.arange(0, MAX_PAGES, layout=layout)
    logical = gl.minimum(first // 16 + ids, gl.maximum(gl.cdiv(end, 16) - 1, 0))
    pages = gl.load(PAGES + logical)
    if MAP_PAGE:
        pages = gl.load(MAP + pages * 16)
    return pages


@g.jit
def _tile(
    K,
    V,
    MAP,
    pages,
    offset,
    first,
    end,
    BN: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAP_PAGE: gl.constexpr,
    MAX_PAGES: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 16], [2, 16], [gl.num_warps(), 1], [1, 0]
    )
    token = offset + gl.arange(0, BN, layout=gl.SliceLayout(1, layout))
    col = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
    pi = gl.minimum(gl.maximum((token - first) // 16, 0), MAX_PAGES - 1)
    page = gl.gather(pages, pi, 0)
    slot = page + (token % 16) if MAP_PAGE else page * 16 + (token % 16)
    if HAS_MAP and not MAP_PAGE:
        slot = gl.load(MAP + slot)
    valid = (token < end) & (slot >= 0)
    pointer = gl.maximum(slot, 0)[:, None] * 256 + col[None, :]
    return gl.load(K + pointer), gl.load(V + pointer), valid


@g.jit
def _partial(
    Q,
    K,
    V,
    PAGES,
    MAP,
    LENGTH,
    KS,
    VS,
    split,
    SCALE: gl.constexpr,
    LEFT: gl.constexpr,
    SPLITS: gl.constexpr,
    BN: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAX_PAGES: gl.constexpr,
    MAP_PAGE: gl.constexpr,
    PREFETCH: gl.constexpr,
):
    mma: gl.constexpr = AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0], [2, 0, 0]]
            if gl.num_warps() == 16
            else [[0, 1, 0], [1, 0, 0], [2, 0, 0]]
        ),
        instr_shape=[16, 16, 32],
        rank=3,
    )
    ql: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 4, 8], [4, gl.num_warps() // 4, 1], [2, 1, 0]
    )
    h = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, ql)))
    r = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(2, ql)))
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, ql)))
    q = gl.load(
        Q
        + (r[None, :, None] // 8) * 2048
        + (h[:, None, None] * 8 + r[None, :, None] % 8) * 64
        + d[None, None, :]
    )
    q = gl.convert_layout(q, gl.DotOperandLayout(0, mma, 8))
    length = gl.load(LENGTH)
    first = gl.maximum(0, length - 8 - LEFT) if LEFT >= 0 else 0
    first = first // BN * BN
    tiles = gl.cdiv(length - first, BN)
    start = first + (split * tiles // SPLITS) * BN
    end = gl.minimum(length, first + ((split + 1) * tiles // SPLITS) * BN)
    pages = _cached_pages(PAGES, MAP, start, end, MAX_PAGES, MAP_PAGE)
    acc = gl.full((4, 64, 64), 0, gl.float32, mma)
    red: gl.constexpr = gl.SliceLayout(2, mma)
    rr = gl.arange(0, 64, layout=gl.SliceLayout(0, red))
    cc = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, mma)))
    maximum = gl.full((4, 64), float("-inf"), gl.float32, red)
    denom = gl.full((4, 64), 0, gl.float32, red)
    scale = SCALE * 1.4426950408889634 * gl.load(KS)
    k0, v0, valid0 = _tile(
        K, V, MAP, pages, start, start, end, BN, HAS_MAP, MAP_PAGE, MAX_PAGES
    )
    k1, v1, valid1 = _tile(
        K, V, MAP, pages, start + BN, start, end, BN, HAS_MAP, MAP_PAGE, MAX_PAGES
    )
    if PREFETCH == 3:
        k2, v2, valid2 = _tile(
            K,
            V,
            MAP,
            pages,
            start + 2 * BN,
            start,
            end,
            BN,
            HAS_MAP,
            MAP_PAGE,
            MAX_PAGES,
        )
    for offset in range(start, end, BN):
        kk = gl.reshape(k0.to(gl.bfloat16), (BN, 4, 64)).permute((1, 2, 0))
        vv = gl.reshape(
            gl.where(valid0[:, None], v0.to(gl.bfloat16), 0.0), (BN, 4, 64)
        ).permute((1, 0, 2))
        valid = valid0
        k0, v0, valid0 = k1, v1, valid1
        if PREFETCH == 3:
            k1, v1, valid1 = k2, v2, valid2
            k2, v2, valid2 = _tile(
                K,
                V,
                MAP,
                pages,
                offset + 3 * BN,
                start,
                end,
                BN,
                HAS_MAP,
                MAP_PAGE,
                MAX_PAGES,
            )
        else:
            k1, v1, valid1 = _tile(
                K,
                V,
                MAP,
                pages,
                offset + 2 * BN,
                start,
                end,
                BN,
                HAS_MAP,
                MAP_PAGE,
                MAX_PAGES,
            )
        kk = gl.convert_layout(kk, gl.DotOperandLayout(1, mma, 8))
        score = gfx1250.wmma(q, kk, gl.full((4, 64, BN), 0, gl.float32, mma)) * scale
        valid = gl.convert_layout(valid, gl.SliceLayout(0, gl.SliceLayout(1, mma)))
        absolute = length - 8 + rr // 8
        pos = offset + cc
        mask = (pos[None, None, :] <= absolute[None, :, None]) & valid[None, None, :]
        if LEFT >= 0:
            mask &= pos[None, None, :] >= absolute[None, :, None] - LEFT
        score = gl.where(mask, score, float("-inf"))
        new_max = gl.maximum(maximum, gl.max(score, 2))
        safe = gl.where(new_max == float("-inf"), 0.0, new_max)
        prob = gl.exp2(score - safe[:, :, None])
        alpha = gl.exp2(maximum - safe)
        denom = denom * alpha + gl.sum(prob, 2)
        acc *= alpha[:, :, None]
        acc = gfx1250.wmma(
            gl.convert_layout(prob.to(gl.bfloat16), gl.DotOperandLayout(0, mma, 8)),
            gl.convert_layout(vv, gl.DotOperandLayout(1, mma, 8)),
            acc,
        )
        maximum = new_max
    result = acc * (gl.load(VS) / gl.where(denom > 0, denom, 1.0))[:, :, None]
    lse = gl.where(denom > 0, maximum + gl.log2(denom), float("-inf"))
    return result, lse


@g.jit
def _publish(P, LSE, FLAGS, task, tag, value, lse, SPLITS: gl.constexpr):
    flat = gl.reshape(value, (16384,))
    a, b, c, d = vector_memory.split4(flat.to(gl.int32, bitcast=True), 4096)
    ids = gl.arange(0, 4096, layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]))
    _store128(P + ((ids * 4 // 64) * SPLITS + task) * 64 + (ids * 4 % 64), a, b, c, d)
    ll: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    rr = gl.arange(0, 256, layout=ll)
    lse = gl.convert_layout(gl.reshape(lse, (256,)), ll)
    scoped.store_words(
        LSE + rr * SPLITS + task, lse.to(gl.int32, bitcast=True), rr >= 0
    )
    scoped.drain_stores()


@g.jit
def _merge(
    P,
    LSE,
    FLAGS,
    STATUS,
    OUT,
    consumer,
    tag,
    SPLITS: gl.constexpr,
    TRACE,
    task_index,
    SCOPED_IO: gl.constexpr = False,
    ASSUME_READY: gl.constexpr = False,
):
    sl: gl.constexpr = _merge_power_of_two(SPLITS)
    if ASSUME_READY:
        ready = gl.full(
            (), True, gl.int1, gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
        )
    else:
        seen, spins = scoped.wait_flag(
            FLAGS + consumer * 32,
            tag,
            100000000,
            MAX_SPINS=4096,
            CLOCK_EVERY=64,
            SHADER_CLOCK=True,
        )
        ready = seen == tag
    if TRACE is not None:
        gl.store(TRACE + consumer * 8 + 3, sync.clock())
    if ready:
        ml: gl.constexpr = gl.BlockedLayout(
            [1, 4], [2, 16], [gl.num_warps(), 1], [1, 0]
        )
        pl: gl.constexpr = gl.BlockedLayout(
            [1, 1], [2, 16], [gl.num_warps(), 1], [1, 0]
        )
        ss = gl.arange(0, sl, layout=gl.SliceLayout(1, pl))
        dd = gl.arange(0, 16, layout=gl.SliceLayout(0, pl))
        pointer = (
            P
            + (consumer * SPLITS + gl.minimum(ss[:, None], SPLITS - 1)) * 64
            + dd[None, :] * 4
        )
        pack: gl.constexpr = min(32, max(1, sl * 16 // (gl.num_warps() * 32)))
        a, b, c, d = _load128(pointer, pack)
        if TRACE is not None:
            gl.store(TRACE + consumer * 8 + 7, sync.clock())
        joined = gl.join(gl.join(a, b), gl.join(c, d)).permute((0, 1, 3, 2))
        val = gl.convert_layout(gl.reshape(joined, (sl, 64)), ml).to(
            gl.float32, bitcast=True
        )
        il: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
        i = gl.arange(0, sl, layout=il)
        lse = scoped.load_words_masked(
            LSE + consumer * SPLITS + gl.minimum(i, SPLITS - 1), i < SPLITS
        ).to(gl.float32, bitcast=True)
        lse = gl.where(i < SPLITS, lse, float("-inf"))
        lse = gl.convert_layout(lse, gl.SliceLayout(1, ml))
        if TRACE is not None:
            gl.store(TRACE + consumer * 8 + 4, sync.clock())
        maximum = gl.max(lse, 0)
        maximum = gl.where(maximum == float("-inf"), 0.0, maximum)
        weight = gl.exp2(lse - maximum)
        total = gl.sum(weight, 0)
        result = gl.sum(val * weight[:, None], 0) / gl.where(total > 0, total, 1.0)
        if TRACE is not None:
            gl.store(TRACE + consumer * 8 + 5, sync.clock())
        dim = gl.arange(0, 64, layout=gl.SliceLayout(0, ml))
        if SCOPED_IO:
            store_bf16_vector(
                OUT
                + (consumer % 64 // 8) * 2048
                + (consumer // 64 * 8 + consumer % 8) * 64,
                result,
                64,
            )
        else:
            gl.store(
                OUT
                + (consumer % 64 // 8) * 2048
                + (consumer // 64 * 8 + consumer % 8) * 64
                + dim,
                result.to(gl.bfloat16),
            )

    gl.store(STATUS + consumer, ready.to(gl.int32))
    if TRACE is not None:
        gl.store(TRACE + consumer * 8 + 6, sync.clock())

    return ready


@g.jit
def _head_tile(
    K,
    V,
    MAP,
    pages,
    offset,
    first,
    end,
    head,
    BN: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAP_PAGE: gl.constexpr,
    MAX_PAGES: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [gl.num_warps(), 1], [1, 0])
    token = offset + gl.arange(0, BN, layout=gl.SliceLayout(1, layout))
    dim = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    pages = gl.convert_layout(pages, gl.SliceLayout(1, layout))
    pi = gl.minimum(gl.maximum((token - first) // 16, 0), MAX_PAGES - 1)
    page = gl.gather(pages, pi, 0)
    slot = page + token % 16 if MAP_PAGE else page * 16 + token % 16
    if HAS_MAP and not MAP_PAGE:
        slot = gl.load(MAP + slot)
    valid = (token < end) & (slot >= 0)
    pointer = gl.maximum(slot, 0)[:, None] * 256 + head * 64 + dim[None, :]
    return gl.load(K + pointer), gl.load(V + pointer), valid


@g.jit
def _window_partial(
    Q,
    K,
    V,
    PAGES,
    MAP,
    LENGTH,
    KS,
    VS,
    head,
    chunk,
    SCALE: gl.constexpr,
    SPLITS: gl.constexpr,
    BN: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAX_PAGES: gl.constexpr,
    MAP_PAGE: gl.constexpr,
):
    mma: gl.constexpr = AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[1, 0], [2, 0], [0, 1]],
        instr_shape=[16, 16, 32],
    )
    load: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0])
    r = gl.arange(0, 64, layout=gl.SliceLayout(1, load))
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, load))
    q = gl.load(
        Q + (r[:, None] // 8) * 2048 + (head * 8 + r[:, None] % 8) * 64 + d[None, :]
    )
    q = gl.convert_layout(q, gl.DotOperandLayout(0, mma, 8))
    length = gl.load(LENGTH)
    first = gl.maximum(0, length - 8 - 2047) // BN * BN
    tiles = gl.cdiv(length - first, BN)
    start = first + chunk * tiles // SPLITS * BN
    end = gl.minimum(length, first + (chunk + 1) * tiles // SPLITS * BN)
    pages = _cached_pages(PAGES, MAP, start, end, MAX_PAGES, MAP_PAGE)
    k0, v0, valid0 = _head_tile(
        K, V, MAP, pages, start, start, end, head, BN, HAS_MAP, MAP_PAGE, MAX_PAGES
    )
    k1, v1, valid1 = _head_tile(
        K, V, MAP, pages, start + BN, start, end, head, BN, HAS_MAP, MAP_PAGE, MAX_PAGES
    )
    row = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    col = gl.arange(0, BN, layout=gl.SliceLayout(0, mma))
    acc = gl.full((64, 64), 0, gl.float32, mma)
    maximum = gl.full((64,), float("-inf"), gl.float32, gl.SliceLayout(1, mma))
    denom = gl.full((64,), 0, gl.float32, gl.SliceLayout(1, mma))
    scale = SCALE * 1.4426950408889634 * gl.load(KS)
    for offset in range(start, end, BN):
        kk = gl.convert_layout(
            k0.to(gl.bfloat16).permute((1, 0)), gl.DotOperandLayout(1, mma, 8)
        )
        vv = gl.convert_layout(
            gl.where(valid0[:, None], v0.to(gl.bfloat16), 0.0),
            gl.DotOperandLayout(1, mma, 8),
        )
        valid = gl.convert_layout(valid0, gl.SliceLayout(0, mma))
        k0, v0, valid0 = k1, v1, valid1
        k1, v1, valid1 = _head_tile(
            K,
            V,
            MAP,
            pages,
            offset + 2 * BN,
            start,
            end,
            head,
            BN,
            HAS_MAP,
            MAP_PAGE,
            MAX_PAGES,
        )
        score = gfx1250.wmma(q, kk, gl.full((64, BN), 0, gl.float32, mma)) * scale
        pos = offset + col
        absolute = length - 8 + row // 8
        mask = (
            (pos[None, :] <= absolute[:, None])
            & (pos[None, :] >= absolute[:, None] - 2047)
            & valid[None, :]
        )
        score = gl.where(mask, score, float("-inf"))
        new_max = gl.maximum(maximum, gl.max(score, 1))
        safe = gl.where(new_max == float("-inf"), 0.0, new_max)
        prob = gl.exp2(score - safe[:, None])
        alpha = gl.exp2(maximum - safe)
        denom = denom * alpha + gl.sum(prob, 1)
        acc *= alpha[:, None]
        acc = gfx1250.wmma(
            gl.convert_layout(prob.to(gl.bfloat16), gl.DotOperandLayout(0, mma, 8)),
            vv,
            acc,
        )
        maximum = new_max
    return acc * (gl.load(VS) / gl.where(denom > 0, denom, 1.0))[:, None], gl.where(
        denom > 0, maximum + gl.log2(denom), float("-inf")
    )


@g.jit
def _window_stage(
    task,
    Q,
    K,
    V,
    PAGES,
    MAP,
    LENGTH,
    KS,
    VS,
    P,
    LSE,
    COUNTERS,
    FLAGS,
    STATUS,
    OUT,
    ctas,
    SCALE: gl.constexpr,
    SPLITS: gl.constexpr,
    BN: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAX_PAGES: gl.constexpr,
    MAP_PAGE: gl.constexpr,
    TRACE,
):
    head = task // SPLITS
    chunk = task % SPLITS
    if TRACE is not None:
        gl.store(TRACE + task * 8, sync.clock())
    tag = _advance_tag(COUNTERS, task)
    value, lse = _window_partial(
        Q,
        K,
        V,
        PAGES,
        MAP,
        LENGTH,
        KS,
        VS,
        head,
        chunk,
        SCALE,
        SPLITS,
        BN,
        HAS_MAP,
        MAX_PAGES,
        MAP_PAGE,
    )
    if TRACE is not None:
        gl.store(TRACE + task * 8 + 1, sync.clock())
    ll: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    row = gl.arange(0, 64, layout=ll)
    lse = gl.convert_layout(lse, ll)
    scoped.store_words(
        LSE + (head * 64 + row) * SPLITS + chunk,
        lse.to(gl.int32, bitcast=True),
        row >= 0,
    )
    pa, pb, pc, pd = vector_memory.split4(
        gl.reshape(value, (4096,)).to(gl.int32, bitcast=True), 1024
    )
    ids = gl.arange(0, 1024, layout=ll)
    # The b128 batch's wait drains both payload and preceding LSE stores.
    _store128(
        P + ((head * 64 + ids * 4 // 64) * SPLITS + chunk) * 64 + ids * 4 % 64,
        pa,
        pb,
        pc,
        pd,
        1024,
    )
    gl.barrier()
    ticket = _advance_tag(COUNTERS, ctas + head)
    if ticket == tag * SPLITS:
        consumer = gl.arange(0, SPLITS, layout=ll)
        scoped.store_words(
            FLAGS + (head * SPLITS + consumer) * 32,
            gl.full_like(consumer, tag),
            consumer >= 0,
        )
    if TRACE is not None:
        gl.store(TRACE + task * 8 + 2, sync.clock())
    seen, spins = scoped.wait_flag(
        FLAGS + task * 32,
        tag,
        100000000,
        MAX_SPINS=4096,
        CLOCK_EVERY=64,
        SHADER_CLOCK=True,
    )
    ready = seen == tag
    if TRACE is not None:
        gl.store(TRACE + task * 8 + 3, sync.clock())
    ROWS: gl.constexpr = 64 // SPLITS
    if ready:
        ml: gl.constexpr = gl.BlockedLayout(
            [1, 1, 4], [1, 2, 16], [SPLITS, 1, gl.num_warps() // SPLITS], [2, 1, 0]
        )
        pl: gl.constexpr = gl.BlockedLayout(
            [1, 1, 1], [1, 2, 16], [SPLITS, 1, gl.num_warps() // SPLITS], [2, 1, 0]
        )
        s = gl.arange(0, SPLITS, layout=gl.SliceLayout(1, gl.SliceLayout(2, pl)))
        r = gl.arange(0, ROWS, layout=gl.SliceLayout(0, gl.SliceLayout(2, pl)))
        d = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, pl)))
        packet = (
            P
            + (
                (head * 64 + chunk * ROWS + r[None, :, None]) * SPLITS
                + s[:, None, None]
            )
            * 64
            + d[None, None, :] * 4
        )
        PACK: gl.constexpr = max(
            1, min(32, SPLITS * ROWS * 16 // (gl.num_warps() * 32))
        )
        a, b, c, d = _load128(packet, PACK)
        if TRACE is not None:
            gl.store(TRACE + task * 8 + 7, sync.clock())
        joined = gl.join(gl.join(a, b), gl.join(c, d)).permute((0, 1, 2, 4, 3))
        values = gl.convert_layout(gl.reshape(joined, (SPLITS, ROWS, 64)), ml).to(
            gl.float32, bitcast=True
        )
        i = gl.arange(0, 64, layout=ll)
        logs = scoped.load_words_masked(
            LSE + (head * 64 + chunk * ROWS + i // SPLITS) * SPLITS + i % SPLITS, i >= 0
        ).to(gl.float32, bitcast=True)
        logs = gl.convert_layout(
            gl.reshape(logs, (ROWS, SPLITS)).permute((1, 0)), gl.SliceLayout(2, ml)
        )
        if TRACE is not None:
            gl.store(TRACE + task * 8 + 4, sync.clock())
        maximum = gl.max(logs, 0)
        maximum = gl.where(maximum == float("-inf"), 0.0, maximum)
        weight = gl.exp2(logs - maximum[None, :])
        total = gl.sum(weight, 0)
        total = gl.convert_layout(total, gl.SliceLayout(1, gl.SliceLayout(0, ml)))
        result = (
            gl.sum(values * weight[:, :, None], 0)
            / gl.where(total > 0, total, 1.0)[:, None]
        )
        if TRACE is not None:
            gl.store(TRACE + task * 8 + 5, sync.clock())
        r = chunk * ROWS + gl.arange(
            0, ROWS, layout=gl.SliceLayout(1, gl.SliceLayout(0, ml))
        )
        dim = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(0, ml)))
        gl.store(
            OUT
            + (r[:, None] // 8) * 2048
            + (head * 8 + r[:, None] % 8) * 64
            + dim[None, :],
            result.to(gl.bfloat16),
        )
    rows = head * 64 + chunk * ROWS + gl.arange(0, ROWS, layout=ll)
    gl.store(STATUS + rows, gl.full_like(rows, ready.to(gl.int32)))
    if TRACE is not None:
        gl.store(TRACE + task * 8 + 6, sync.clock())


@g.jit
def attention_stage(
    task_index,
    ctas,
    Q,
    K,
    V,
    PAGES,
    MAP,
    LENGTH,
    KS,
    VS,
    P,
    LSE,
    COUNTERS,
    FLAGS,
    STATUS,
    OUT,
    SCALE: gl.constexpr,
    LEFT: gl.constexpr,
    SPLITS: gl.constexpr,
    BN: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAX_PAGES: gl.constexpr = 256,
    MAP_PAGE: gl.constexpr = False,
    PREFETCH: gl.constexpr = 2,
    TRACE=None,
):
    if LEFT >= 0 and (SPLITS == 4 or SPLITS == 8):
        gl.static_assert(gl.num_warps() == 8, "Grouped sliding requires eight waves")
        if task_index < 4 * SPLITS:
            _window_stage(
                task_index,
                Q,
                K,
                V,
                PAGES,
                MAP,
                LENGTH,
                KS,
                VS,
                P,
                LSE,
                COUNTERS,
                FLAGS,
                STATUS,
                OUT,
                ctas,
                SCALE,
                SPLITS,
                BN,
                HAS_MAP,
                MAX_PAGES,
                MAP_PAGE,
                TRACE,
            )
    else:
        if TRACE is not None:
            gl.store(TRACE + task_index * 8, sync.clock())
        tag = _advance_tag(COUNTERS, task_index)
        for task in range(task_index, SPLITS, ctas):
            value, lse = _partial(
                Q,
                K,
                V,
                PAGES,
                MAP,
                LENGTH,
                KS,
                VS,
                task,
                SCALE,
                LEFT,
                SPLITS,
                BN,
                HAS_MAP,
                MAX_PAGES,
                MAP_PAGE,
                PREFETCH,
            )
            if TRACE is not None:
                gl.store(TRACE + task_index * 8 + 1, sync.clock())
            _publish(P, LSE, FLAGS, task, tag, value, lse, SPLITS)
        group = task_index // 8
        groups = gl.cdiv(ctas, 8)
        members = gl.minimum(8, ctas - group * 8)
        ticket = _advance_tag(COUNTERS, ctas + group)
        if ticket == tag * members:
            top = _advance_tag(COUNTERS, ctas + groups)
            if top == tag * groups:
                scoped.publish_flags(FLAGS, tag, 256)
        if TRACE is not None:
            gl.store(TRACE + task_index * 8 + 2, sync.clock())
        for consumer in range(task_index, 256, ctas):
            if TRACE is not None:
                if consumer >= ctas:
                    gl.store(TRACE + consumer * 8 + 2, sync.clock())
            _merge(P, LSE, FLAGS, STATUS, OUT, consumer, tag, SPLITS, TRACE, task_index)


@g.jit
def _standalone(
    Q,
    K,
    V,
    PAGES,
    MAP,
    LENGTH,
    KS,
    VS,
    P,
    LSE,
    COUNTERS,
    FLAGS,
    STATUS,
    OUT,
    SCALE: gl.constexpr,
    LEFT: gl.constexpr,
    SPLITS: gl.constexpr,
    BN: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAX_PAGES: gl.constexpr = 256,
    MAP_PAGE: gl.constexpr = False,
    PREFETCH: gl.constexpr = 2,
    TRACE=None,
):
    attention_stage(
        gl.program_id(0),
        gl.num_programs(0),
        Q,
        K,
        V,
        PAGES,
        MAP,
        LENGTH,
        KS,
        VS,
        P,
        LSE,
        COUNTERS,
        FLAGS,
        STATUS,
        OUT,
        SCALE,
        LEFT,
        SPLITS,
        BN,
        HAS_MAP,
        MAX_PAGES,
        MAP_PAGE,
        PREFETCH,
        TRACE,
    )


def attention(
    q,
    k,
    v,
    page_table,
    length,
    k_descale,
    v_descale,
    *,
    context,
    softmax_scale=0.125,
    window_left=-1,
    mapping=None,
    mapping_page_aligned=False,
    workspace=None,
    ctas=192,
    block_n=64,
    waves=8,
    prefetch=2,
    sliding_splits=8,
):
    """S2 only; length is8..context. Caller guarantees valid active page/map entries."""
    assert q.shape == (8, 32, 64) and q.dtype == torch.bfloat16 and q.is_contiguous()
    assert k.shape == v.shape and k.shape[1:] == (16, 4, 64)
    assert (
        k.dtype == v.dtype == torch.float8_e4m3fn
        and k.is_contiguous()
        and v.is_contiguous()
    )
    assert window_left in (-1, 2047) and 8 <= context <= 131072 and waves in (8, 16)
    assert block_n in (32, 64, 128) and 64 <= ctas <= 192 and prefetch in (2, 3)
    assert (
        page_table.numel() >= (context + 15) // 16
        and page_table.dtype == torch.int32
        and page_table.is_contiguous()
    )
    assert (
        length.numel() == 1 and length.dtype == torch.int32 and length.is_contiguous()
    )
    tensors = [k, v, page_table, length, k_descale, v_descale]
    if mapping is not None:
        assert mapping.dtype in (torch.int32, torch.int64) and mapping.is_contiguous()
        tensors.append(mapping)
    assert not mapping_page_aligned or mapping is not None
    assert q.is_cuda and all(t.device == q.device for t in tensors)
    assert (
        k_descale.numel() == v_descale.numel() == 1
        and k_descale.dtype == v_descale.dtype == torch.float32
    )
    if workspace is None:
        workspace = make_workspace(
            q.device,
            context=context,
            window_left=window_left,
            ctas=ctas,
            block_n=block_n,
            sliding_splits=sliding_splits,
        )
    if window_left >= 0 and workspace.splits in (4, 8):
        assert waves == 8, "Use sliding_splits=0 for the legacy 16-wave path"
    assert workspace.ctas == ctas and workspace.partial.shape == (
        workspace.splits,
        256,
        64,
    )
    assert workspace.lse.shape == (workspace.splits, 256) and workspace.flags.shape == (
        256,
        32,
    )
    assert workspace.status.shape == (256,) and workspace.counters.shape == (
        ctas + triton.cdiv(ctas, 8) + 1,
        32,
    )
    assert all(
        t.device == q.device and t.dtype == torch.int32 and t.is_contiguous()
        for t in (
            workspace.partial,
            workspace.lse,
            workspace.counters,
            workspace.flags,
            workspace.status,
        )
    )
    out = torch.empty_like(q)
    _standalone[(ctas,)](
        q,
        k,
        v,
        page_table,
        mapping,
        length,
        k_descale,
        v_descale,
        workspace.partial,
        workspace.lse,
        workspace.counters,
        workspace.flags,
        workspace.status,
        out,
        softmax_scale,
        window_left,
        workspace.splits,
        block_n,
        mapping is not None,
        workspace.pages_per_chunk,
        mapping_page_aligned,
        prefetch,
        num_warps=waves,
        enable_fp_fusion=False,
    )
    return out

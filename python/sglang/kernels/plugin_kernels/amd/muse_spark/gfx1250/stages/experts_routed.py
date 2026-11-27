"""A16W4 S5 task types using S4's BLOCK_M16 native hit table.

Every task contains either W13+SwiGLU or W2. Only the sixteen hit slots of its
expert/M-block enter WMMA; sentinel64 rows are masked/zero. Intermediates and
outputs are BF16-rounded values in FP32 words indexed by original hit slot.
W13 and W2 share no activation cache or per-expert vectors across tasks.
Private128-byte completion flags belong to each M-block/N-tile consumer.

Only serial-stream launches may reuse state. Check all CTA statuses; discard
state/output after any timeout. Every poll is bounded. No physical peer APIs.
"""

import msgspec
import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.vector_memory import (
    _load_asm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    experts_aligned,
    experts_native_core,
    experts_tracked,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts import (
    _fp4_upcast,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts_valu import (
    valu_task,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import _mma


@g.jit
def _weights(
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
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 8, 1], [1, 4, 8], [S, 1, gl.num_warps() // S], [1, 2, 0]
    )
    ss = gl.arange(0, S, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    kk = gl.arange(0, BK // 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    nn = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    offset = ss[:, None, None] * (4096 // S // 2) + step * (BK // 2) + kk[None, :, None]
    row = tile * BN + nn
    packed = gl.load(
        W + expert * ROWS * 2048 + row[None, None, :] * 2048 + offset,
        cache_modifier=".cs",
    )
    scales = gl.load(
        Scale + expert * ROWS * 128 + row[None, None, :] * 128 + offset // 16
    )
    return packed, scales


@g.jit
def _masked_packets(ptr, valid):
    """Reuse audited b128 EXEC masks, LOADcnt and XCNT; padded rows issue no loads."""
    asm: gl.constexpr = _load_asm(1, 16)
    a, b, c, d, saved = gl.inline_asm_elementwise(
        asm[0],
        asm[1],
        [ptr, valid.to(gl.int32)],
        dtype=(gl.int32,) * 5,
        is_pure=False,
        pack=16,
    )
    return a, b, c, d


@g.jit
def _cache_hits(X, Sorted, block, SECOND: gl.constexpr):
    shared = gl.allocate_shared_memory(
        gl.bfloat16, [4, 16384], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 4], [1, 0])
    for chunk in gl.static_range(2):
        if SECOND:
            q = gl.arange(0, 8192, layout=gl.BlockedLayout([1], [32], [16], [0]))
            slots = gl.load(Sorted + block * 16 + chunk * 8 + q // 1024)
            a, b, c, d = _masked_packets(X + slots * 4096 + (q % 1024) * 4, slots < 64)
            value = (
                gl.join(gl.join(a, b), gl.join(c, d))
                .permute((0, 2, 1))
                .reshape((32768,))
            )
            value = value.to(gl.float32, bitcast=True).to(gl.bfloat16)
        else:
            i = gl.arange(0, 32768, layout=gl.BlockedLayout([8], [32], [16], [0]))
            slots = gl.load(Sorted + block * 16 + chunk * 8 + i // 4096)
            value = gl.load(X + (slots // 8) * 4096 + i % 4096, slots < 64, other=0)
        value = value.reshape((8, 4, 1024)).permute((1, 0, 2)).reshape((4, 8192))
        value = gl.convert_layout(value, layout)
        shared.slice(chunk * 8192, 8192, dim=1).store(value)
    return shared


@g.jit
def _hit_tile(cached, step, BN: gl.constexpr, BK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 4], [1, 0])
    j = gl.arange(0, 16 * BK, layout=gl.SliceLayout(0, layout))
    index = (
        gl.full((4, 16 * BK), 0, gl.int32, layout)
        + (j[None, :] // BK) * 1024
        + step * BK
        + j[None, :] % BK
    )
    value = cached.gather(index, 1).reshape((4, 16, BK))
    return gl.convert_layout(value, gl.DotOperandLayout(0, _mma(16, BN), 8))


@g.jit
def _routed_gemm(
    X,
    Sorted,
    W,
    Scale,
    block,
    expert,
    tile,
    ROWS: gl.constexpr,
    SECOND: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    ALIGNED: gl.constexpr = False,
    TRACKED: gl.constexpr = False,
    CORE_K: gl.constexpr = 0,
):
    S: gl.constexpr = gl.num_warps() // (BN // 16)
    mma: gl.constexpr = _mma(gl.num_warps(), BN)
    acc = gl.full((S, 16, BN), 0, gl.float32, mma)
    p0, s0 = (
        experts_tracked.weight_packets
        if TRACKED
        else experts_aligned.weight_packets
        if ALIGNED
        else _weights
    )(W, Scale, expert, tile, 0, ROWS, BN, BK, S)
    p1, s1 = (
        experts_tracked.weight_packets
        if TRACKED
        else experts_aligned.weight_packets
        if ALIGNED
        else _weights
    )(W, Scale, expert, tile, 1, ROWS, BN, BK, S)
    p2, s2 = (
        experts_tracked.weight_packets
        if TRACKED
        else experts_aligned.weight_packets
        if ALIGNED
        else _weights
    )(W, Scale, expert, tile, 2, ROWS, BN, BK, S)
    for step in range(4096 // S // BK):
        a = _hit_tile(X, step, BN, BK)
        if TRACKED:
            b = experts_tracked.upcast(p0, s0, BN, BK, S)
        elif ALIGNED:
            b = experts_aligned.upcast(p0, s0)
        else:
            b = _fp4_upcast(p0, s0, BN, BK, S, "direct")
        acc = gfx1250.wmma(a, b, acc)
        p0 = p1
        s0 = s1
        p1 = p2
        s1 = s2
        if step + 3 < 4096 // S // BK:
            p2, s2 = (
                experts_tracked.weight_packets
                if TRACKED
                else experts_aligned.weight_packets
                if ALIGNED
                else _weights
            )(W, Scale, expert, tile, step + 3, ROWS, BN, BK, S)
    return gl.sum(acc, 0)


@g.jit
def _store_slots(Out, Sorted, block, tile, value, BN: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [16, 1], [1, 0])
    row = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    col = gl.arange(0, BN, layout=gl.SliceLayout(0, layout))
    slots = gl.load(Sorted + block * 16 + row)
    value = gl.convert_layout(value, layout).to(gl.bfloat16).to(gl.float32)
    scoped.store_words(
        Out + slots[:, None] * 4096 + tile * BN + col[None, :],
        value.to(gl.int32, bitcast=True),
        slots[:, None] < 64,
    )


@g.jit
def w13_task(
    X,
    Sorted,
    W,
    S,
    Inter,
    Arrivals,
    Flags,
    block,
    expert,
    tile,
    tag,
    BN: gl.constexpr = 64,
    BK: gl.constexpr = 64,
    ALIGNED: gl.constexpr = False,
    TRACKED: gl.constexpr = False,
    CORE_K: gl.constexpr = 0,
):
    if CORE_K:
        value = experts_native_core.gemm(
            X, Sorted, W, S, block, expert, tile, CORE_K, False
        )
    else:
        cached = _cache_hits(X, Sorted, block, False)
        gate = _routed_gemm(
            cached,
            Sorted,
            W,
            S,
            block,
            expert,
            tile,
            8192,
            False,
            BN,
            BK,
            ALIGNED,
            TRACKED,
        )
        up = _routed_gemm(
            cached,
            Sorted,
            W,
            S,
            block,
            expert,
            tile + 4096 // BN,
            8192,
            False,
            BN,
            BK,
            ALIGNED,
            TRACKED,
        )
        value = gate / (1.0 + gl.exp(-gate)) * up
    _store_slots(Inter, Sorted, block, tile, value, BN)
    scoped.drain_stores()
    ticket = _advance_tag(Arrivals, block)
    if ticket % (4096 // BN) == 0:
        scoped.publish_flags(Flags + block * (4096 // BN) * 32, tag, 4096 // BN)


@g.jit
def w2_task(
    Inter,
    Sorted,
    W,
    S,
    Out,
    Flags,
    block,
    expert,
    tile,
    tag,
    BN: gl.constexpr = 64,
    BK: gl.constexpr = 64,
    ALIGNED: gl.constexpr = False,
    TRACKED: gl.constexpr = False,
    CORE_K: gl.constexpr = 0,
):
    seen, polls = scoped.wait_flag(
        Flags + (block * (4096 // BN) + tile) * 32, tag, 100_000_000, SHADER_CLOCK=True
    )
    if seen == tag:
        if CORE_K:
            value = experts_native_core.gemm(
                Inter, Sorted, W, S, block, expert, tile, CORE_K, True
            )
        else:
            cached = _cache_hits(Inter, Sorted, block, True)
            value = _routed_gemm(
                cached,
                Sorted,
                W,
                S,
                block,
                expert,
                tile,
                4096,
                True,
                BN,
                BK,
                ALIGNED,
                TRACKED,
            )
        _store_slots(Out, Sorted, block, tile, value, BN)
    return seen == tag


@g.jit
def routed_stage(
    task_index,
    ctas,
    X,
    Sorted,
    Experts,
    NumPost,
    W13,
    S13,
    W2,
    S2,
    Inter,
    Out,
    Epochs,
    Arrivals,
    Flags,
    Status,
    SKIP: gl.constexpr = -1,
    H_VALU: gl.constexpr = 0,
    ALIGNED: gl.constexpr = False,
    TRACKED: gl.constexpr = False,
    CORE_K: gl.constexpr = 0,
):
    tag = _advance_tag(Epochs, task_index)
    tasks = (gl.load(NumPost) // 16) * 64
    ok = gl.full((), 1, gl.int32, gl.BlockedLayout([1], [32], [16], [0]))
    for task in range(task_index, 2 * tasks, ctas):
        index = task % tasks
        block = index // 64
        tile = index % 64
        expert = gl.load(Experts + block)
        if H_VALU > 0:
            rr = gl.arange(0, 16, layout=gl.BlockedLayout([1], [32], [16], [0]))
            hits = gl.sum((gl.load(Sorted + block * 16 + rr) < 64).to(gl.int32), 0)
            previous = gl.load(Experts + block - 1, block > 0, other=-1)
            following = gl.load(Experts + block + 1, block + 1 < tasks // 64, other=-1)
            use_valu = (hits <= H_VALU) & (previous != expert) & (following != expert)
        else:
            hits = 0
            use_valu = False
        if task < tasks:
            if index != SKIP:
                if use_valu:
                    valu_task(
                        X,
                        Sorted,
                        W13,
                        S13,
                        Inter,
                        Inter,
                        Arrivals,
                        Flags,
                        block,
                        expert,
                        tile,
                        tag,
                        hits,
                        False,
                        H_VALU,
                    )
                else:
                    w13_task(
                        X,
                        Sorted,
                        W13,
                        S13,
                        Inter,
                        Arrivals,
                        Flags,
                        block,
                        expert,
                        tile,
                        tag,
                        ALIGNED=ALIGNED,
                        TRACKED=TRACKED,
                        CORE_K=CORE_K,
                    )
        else:
            if use_valu:
                ready = valu_task(
                    Inter,
                    Sorted,
                    W2,
                    S2,
                    Inter,
                    Out,
                    Arrivals,
                    Flags,
                    block,
                    expert,
                    tile,
                    tag,
                    hits,
                    True,
                    H_VALU,
                )
            else:
                ready = w2_task(
                    Inter,
                    Sorted,
                    W2,
                    S2,
                    Out,
                    Flags,
                    block,
                    expert,
                    tile,
                    tag,
                    ALIGNED=ALIGNED,
                    TRACKED=TRACKED,
                    CORE_K=CORE_K,
                )
            ok = ok & ready
    gl.store(Status + task_index, ok)


@g.jit
def _standalone(
    X,
    Sorted,
    Experts,
    NumPost,
    W13,
    S13,
    W2,
    S2,
    Inter,
    Out,
    Epochs,
    Arrivals,
    Flags,
    Status,
    SKIP: gl.constexpr = -1,
    H_VALU: gl.constexpr = 0,
    ALIGNED: gl.constexpr = False,
    TRACKED: gl.constexpr = False,
    CORE_K: gl.constexpr = 0,
):
    routed_stage(
        gl.program_id(0),
        gl.num_programs(0),
        X,
        Sorted,
        Experts,
        NumPost,
        W13,
        S13,
        W2,
        S2,
        Inter,
        Out,
        Epochs,
        Arrivals,
        Flags,
        Status,
        SKIP,
        H_VALU,
        ALIGNED,
        TRACKED,
        CORE_K,
    )


class RoutedWorkspace(msgspec.Struct):
    inter: torch.Tensor
    output: torch.Tensor
    epochs: torch.Tensor
    arrivals: torch.Tensor
    flags: torch.Tensor
    status: torch.Tensor


def make_workspace(device):
    """Allocate serial-stream scratch before capture; row64 stays zero."""
    assert torch.cuda.get_device_properties(device).multi_processor_count >= 192
    return RoutedWorkspace(
        torch.zeros((65, 4096), device=device, dtype=torch.float32),
        torch.empty((64, 4096), device=device, dtype=torch.float32),
        torch.zeros((192, 32), device=device, dtype=torch.int32),
        torch.zeros((64, 32), device=device, dtype=torch.int32),
        torch.zeros((64, 64, 32), device=device, dtype=torch.int32),
        torch.empty(192, device=device, dtype=torch.int32),
    )


def experts(
    x,
    sorted_ids,
    expert_ids,
    num_post,
    w13,
    s13,
    w2,
    s2,
    ws,
    *,
    skip=-1,
    h_valu=0,
    aligned=False,
    tracked=False,
    core_k=0,
):
    """Consume the local64/BLOCK_M16 table; all optimization switches are WIP.

    Defaults preserve WMMA. tracked/core_k require the validated Triton3.9 stack.
    Check all192 statuses before using outputs.
    """
    if tracked and triton.__version__.split(".")[:2] != ["3", "9"]:
        raise RuntimeError(
            "experimental tracked S5 requires Triton3.9; integrated Triton3.8 checks failed"
        )
    assert core_k in (0, 64, 128)
    assert not core_k or tracked
    assert h_valu in (0, 1, 2, 4, 8)
    assert not ((aligned or tracked) and h_valu)
    assert not (aligned and tracked)
    assert x.shape == (8, 4096) and x.dtype == torch.bfloat16
    assert w13.shape == (64, 8192, 2048) and s13.shape == (64, 8192, 128)
    assert w2.shape == (64, 4096, 2048) and s2.shape == (64, 4096, 128)
    assert (
        sorted_ids.shape == (1024,)
        and expert_ids.shape == (64,)
        and num_post.shape == (1,)
    )
    assert all(
        t.is_contiguous() and t.device == x.device
        for t in (x, sorted_ids, expert_ids, num_post, w13, s13, w2, s2)
    )
    assert all(t.dtype == torch.uint8 for t in (w13, s13, w2, s2))
    assert all(t.dtype == torch.int32 for t in (sorted_ids, expert_ids, num_post))
    return _standalone[(192,)](
        x,
        sorted_ids,
        expert_ids,
        num_post,
        w13,
        s13,
        w2,
        s2,
        ws.inter,
        ws.output,
        ws.epochs,
        ws.arrivals,
        ws.flags,
        ws.status,
        SKIP=skip,
        H_VALU=h_valu,
        ALIGNED=aligned,
        TRACKED=tracked,
        CORE_K=core_k,
        num_warps=16,
        enable_fp_fusion=False,
    )

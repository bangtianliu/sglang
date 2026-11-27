"""S4 joint BF16 router/pooler, deterministic sigmoid top8 and local slot table.

One192-CTA launch; router high/low accumulators stay FP32, pooler rounds to
BF16 before RMSNorm. Equal biased scores choose the lowest expert id, matching
SGLang's tl.argmax. Local64 experts use BLOCK_M16, ascending routed-slot order
and sentinel64. Serial-stream workspace; discard outputs/state on any timeout.
"""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped, sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_tile_words,
    store_bf16_vector,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    StageState,
    delayed_point,
    handoff,
    make_state,
    store_projection,
    validate_state,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import (
    load_input,
    skinny_tile,
)


class S4Workspace(msgspec.Struct):
    projection: torch.Tensor
    state: StageState
    ids: torch.Tensor
    scores: torch.Tensor
    pooled: torch.Tensor
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_post: torch.Tensor


def make_workspace(device):
    return S4Workspace(
        torch.empty((8, 4608), device=device, dtype=torch.int32),
        make_state(device, ctas=192, phases=2),
        torch.empty((8, 8), device=device, dtype=torch.int32),
        torch.empty((8, 8), device=device, dtype=torch.float32),
        torch.empty((8, 4096), device=device, dtype=torch.bfloat16),
        torch.empty(1024, device=device, dtype=torch.int32),
        torch.empty(64, device=device, dtype=torch.int32),
        torch.empty(1, device=device, dtype=torch.int32),
    )


@g.jit
def _finish_row(
    row,
    P,
    Bias,
    Norm,
    Scores,
    Ids,
    Y,
    Trace=None,
    SCOPED_IO: gl.constexpr = False,
    POOL: gl.constexpr = True,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    n = gl.arange(0, 256, layout=layout)
    high = scoped.load_words_masked(P + row * 4608 + 4096 + n, n >= 0).to(
        gl.float32, bitcast=True
    )
    low = scoped.load_words_masked(P + row * 4608 + 4352 + n, n >= 0).to(
        gl.float32, bitcast=True
    )
    raw = 1.0 / (1.0 + gl.exp(-(high + low) * 0.17677669529663687))
    biased = raw + gl.load(Bias + n)
    if Trace is not None:
        gl.store(Trace + 3, sync.clock())
    slots = gl.arange(0, 8, layout=layout)
    selected = gl.full((8,), 0, gl.float32, layout)
    ids = gl.full((8,), 0, gl.int32, layout)
    total = gl.sum(selected, 0)
    for j in gl.static_range(8):
        maximum = gl.max(biased, 0)
        index = gl.min(gl.where(biased == maximum, n, 2147483647), 0)
        value = gl.sum(gl.where(n == index, raw, 0.0), 0)
        total += value
        ids = gl.where(slots == j, index, ids)
        selected = gl.where(slots == j, value, selected)
        biased = gl.where(n == index, -float("inf"), biased)
    scoped.store_words(Ids + row * 8 + slots, ids, slots >= 0)
    if SCOPED_IO:
        scoped.store_words(
            Scores.to(gl.pointer_type(gl.int32)) + row * 8 + slots,
            (selected / (total + 1e-15)).to(gl.int32, bitcast=True),
            slots >= 0,
        )
    else:
        gl.store(Scores + row * 8 + slots, selected / (total + 1e-15))
    if Trace is not None:
        gl.store(Trace + 4, sync.clock())
    if POOL:
        d = gl.arange(0, 4096, layout=layout)
        value = (
            load_tile_words(P + row * 4608 + d, d >= 0, 4096)
            .to(gl.float32, bitcast=True)
            .to(gl.bfloat16)
            .to(gl.float32)
        )
        if Trace is not None:
            gl.store(Trace + 5, sync.clock())
        value *= gl.rsqrt(gl.sum(value * value, 0) / 4096 + 1e-5)
        value *= gl.load(Norm + d).to(gl.float32)
        if SCOPED_IO:
            store_bf16_vector(Y + row * 4096, value, 4096)
        else:
            gl.store(Y + row * 4096 + d, value.to(gl.bfloat16))
        if Trace is not None:
            gl.store(Trace + 6, sync.clock())


@g.jit
def _local_slots(Ids, Sorted, Experts, NumPost, LOCAL_START: gl.constexpr, Trace=None):
    # One owner reads the64 ids, matching the native small-alignment launch.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [16, 1], [1, 0])
    e = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    j = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    ids = scoped.load_words_masked(Ids + j, j >= 0) - LOCAL_START
    counts = gl.sum((ids[None, :] == e[:, None]).to(gl.int32), 1)
    blocks = gl.cdiv(counts, 16)
    column_blocks = gl.convert_layout(blocks, gl.SliceLayout(0, layout))
    offset = (
        gl.sum(gl.where(j[None, :] < e[:, None], column_blocks[None, :], 0), 1) * 16
    )
    ranks = gl.full((64,), 0, gl.int32, gl.SliceLayout(1, layout))
    if Trace is not None:
        gl.store(Trace + 8, sync.clock())
    for slot in range(64):
        expert = gl.sum(gl.where(j == slot, ids, 0), 0)
        hit = e == expert
        scoped.store_words(Sorted + offset + ranks, gl.full_like(e, slot), hit)
        ranks += hit.to(gl.int32)
    if Trace is not None:
        gl.store(Trace + 9, sync.clock())
    # Eight tokens, unique top8 ids/token: an expert has at most8 slots,
    # so its padded16-row range occupies exactly one block when nonempty.
    scoped.store_words(Experts + offset // 16, e, counts > 0)
    gl.store(NumPost, gl.sum(blocks, 0) * 16)
    if Trace is not None:
        gl.store(Trace + 10, sync.clock())


@g.jit
def s4_stage(
    task_index,
    ctas,
    X,
    W,
    Bias,
    Norm,
    P,
    Epochs,
    Arrivals,
    Flags,
    Status,
    Scores,
    Ids,
    Y,
    Sorted,
    Experts,
    NumPost,
    LOCAL_START: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    TRACE=None,
):
    gl.static_assert(ctas == 192)
    if task_index != SKIP:
        if TRACE is not None:
            gl.store(TRACE + task_index * 16, sync.clock())
        tag = _advance_tag(Epochs, task_index)
        delayed_point(Seed, task_index, 0)
        if task_index == 0:
            i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [16], [0]))
            scoped.store_words(Sorted + i, gl.full_like(i, 64), i >= 0)
            b = gl.arange(0, 64, layout=gl.BlockedLayout([1], [32], [16], [0]))
            scoped.store_words(Experts + b, gl.full_like(b, -1), b >= 0)
        if TRACE is not None:
            gl.store(TRACE + task_index * 16 + 12, sync.clock())
        if task_index < 144:
            cached = load_input(X, 8192, 8, 64)
            value = skinny_tile(
                W, cached, task_index, 4608, 8192, 32, 64, 8, 8, 2, ".cs", False
            )
            store_projection(P, task_index, value, 4608, 32, False)
        if TRACE is not None:
            gl.store(TRACE + task_index * 16 + 1, sync.clock())
        delayed_point(Seed, task_index, 1)
        ready = handoff(Arrivals, Flags, tag, 0, task_index, 192)
        if TRACE is not None:
            gl.store(TRACE + task_index * 16 + 2, sync.clock())
        if ready and task_index < 8:
            _finish_row(
                task_index,
                P,
                Bias,
                Norm,
                Scores,
                Ids,
                Y,
                TRACE + task_index * 16 if TRACE is not None else None,
            )
        delayed_point(Seed, task_index, 2)
        joined = handoff(Arrivals + 32, Flags + 192 * 32, tag, 0, task_index, 192)
        if TRACE is not None:
            gl.store(TRACE + task_index * 16 + 7, sync.clock())
        if ready and joined and task_index == 0:
            _local_slots(
                Ids,
                Sorted,
                Experts,
                NumPost,
                LOCAL_START,
                TRACE + task_index * 16 if TRACE is not None else None,
            )
        gl.store(Status + task_index, (ready & joined).to(gl.int32))
        if TRACE is not None:
            gl.store(TRACE + task_index * 16 + 11, sync.clock())


@g.jit
def _standalone(
    X,
    W,
    Bias,
    Norm,
    P,
    Epochs,
    Arrivals,
    Flags,
    Status,
    Scores,
    Ids,
    Y,
    Sorted,
    Experts,
    NumPost,
    LOCAL_START: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    TRACE=None,
):
    s4_stage(
        gl.program_id(0),
        192,
        X,
        W,
        Bias,
        Norm,
        P,
        Epochs,
        Arrivals,
        Flags,
        Status,
        Scores,
        Ids,
        Y,
        Sorted,
        Experts,
        NumPost,
        LOCAL_START,
        SKIP,
        Seed,
        TRACE,
    )


def routing(x, weight, bias, norm, *, rank=0, workspace=None, delay_seed=None):
    assert x.shape == (8, 8192) and weight.shape == (4608, 8192)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert bias.shape == (256,) and bias.dtype == torch.float32
    assert norm.shape == (4096,) and norm.dtype in (torch.bfloat16, torch.float32)
    assert 0 <= rank < 4
    assert all(
        t.device == x.device and t.is_contiguous() for t in (x, weight, bias, norm)
    )
    if workspace is None:
        workspace = make_workspace(x.device)
    s = workspace.state
    validate_state(s, x.device, ctas=192, phases=2)
    for tensor, shape, dtype in (
        (workspace.projection, (8, 4608), torch.int32),
        (workspace.ids, (8, 8), torch.int32),
        (workspace.scores, (8, 8), torch.float32),
        (workspace.pooled, (8, 4096), torch.bfloat16),
        (workspace.sorted_ids, (1024,), torch.int32),
        (workspace.expert_ids, (64,), torch.int32),
        (workspace.num_post, (1,), torch.int32),
    ):
        assert (
            tensor.shape == shape
            and tensor.dtype == dtype
            and tensor.device == x.device
            and tensor.is_contiguous()
        )
    _standalone[(192,)](
        x,
        weight,
        bias,
        norm,
        workspace.projection,
        s.epochs,
        s.arrivals,
        s.flags,
        s.status,
        workspace.scores,
        workspace.ids,
        workspace.pooled,
        workspace.sorted_ids,
        workspace.expert_ids,
        workspace.num_post,
        rank * 64,
        Seed=delay_seed,
        num_warps=16,
        enable_fp_fusion=False,
    )
    return (
        workspace.scores,
        workspace.ids,
        workspace.pooled,
        workspace.sorted_ids,
        workspace.expert_ids,
        workspace.num_post,
    )

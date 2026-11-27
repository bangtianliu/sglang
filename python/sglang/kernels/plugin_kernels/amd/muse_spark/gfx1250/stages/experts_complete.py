"""Four-wave S5 in one bounded GPU-local launch, four virtual EP ranks.

W13, W2, post-expert norm, original-slot weighted sum, and BF16-partial one-shot
all-reduce share a measured resident grid. Reuse workspace on one serial stream;
discard outputs and workspace after any false status. No physical peer access.
"""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective_fast import (
    oneshot,
    oneshot_plan,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_tile_words,
    store_tile_words,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts_async import (
    expert_tile,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)


@g.jit
def _norm(Y, Gamma, Scaled, Inv, hit, INV_STRIDE: gl.constexpr = 1):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    i = gl.arange(0, 2048, layout=layout)
    bits = load_tile_words(
        Y.to(gl.pointer_type(gl.int32)) + hit * 2048 + i, i >= 0, 2048
    )
    lo = (bits & 65535).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    hi = (bits >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    y = gl.join(lo, hi).reshape((4096,)).to(gl.float32)
    yl: gl.constexpr = y.type.layout
    d = gl.arange(0, 4096, layout=yl)
    gamma = gl.load(Gamma + d)
    effective = (gamma.to(gl.float32) + 1.0).to(gamma.dtype).to(gl.float32)
    y = (y * effective).to(gl.bfloat16).to(gl.float32)
    inv = 1.0 / gl.sqrt(gl.sum(y * y, 0) / 4096.0 + 1e-8)
    store_tile_words(
        Scaled + hit * 4096 + d, y.to(gl.int32, bitcast=True), d >= 0, 4096
    )
    scoped.publish(Inv + hit * INV_STRIDE, inv.to(gl.int32, bitcast=True))


@g.jit
def _sum(
    Ids,
    Scores,
    Scaled,
    Inv,
    Weighted,
    Partial,
    rank,
    row,
    INV_STRIDE: gl.constexpr = 1,
    SCOPED_IO: gl.constexpr = False,
):
    layout: gl.constexpr = gl.BlockedLayout([2], [32], [4], [0])
    d = gl.arange(0, 4096, layout=layout)
    acc = gl.full((4096,), 0, gl.float32, layout)
    for slot in gl.static_range(8):
        hit = row * 8 + slot
        expert = gl.load(Ids + hit, cache_modifier=".cg" if SCOPED_IO else "")
        if (expert >= rank * 64) & (expert < (rank + 1) * 64):
            y = load_tile_words(Scaled + (rank * 64 + hit) * 4096 + d, d >= 0, 4096).to(
                gl.float32, bitcast=True
            )
            inv = scoped.load_flag(Inv + (rank * 64 + hit) * INV_STRIDE).to(
                gl.float32, bitcast=True
            )
            score = gl.load(Scores + hit, cache_modifier=".cg" if SCOPED_IO else "")
            acc += (score * inv) * y
    store_tile_words(
        Weighted + (rank * 8 + row) * 4096 + d,
        acc.to(gl.int32, bitcast=True),
        d >= 0,
        4096,
    )
    lo, hi = gl.split(acc.to(gl.bfloat16).reshape((2048, 2)))
    bits = lo.to(gl.uint16, bitcast=True).to(gl.uint32) | (
        hi.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )
    j = gl.arange(0, 2048, layout=bits.type.layout)
    store_tile_words(
        Partial.to(gl.pointer_type(gl.int32)) + (rank * 8 + row) * 2048 + j,
        bits.to(gl.int32),
        j >= 0,
        2048,
    )


@g.jit
def stage(
    task_index,
    X,
    Ids,
    Scores,
    Sorted,
    Experts,
    NumPost,
    W13,
    S13,
    W2,
    S2,
    Gamma,
    Inter,
    Y,
    Scaled,
    Inv,
    Weighted,
    Partial,
    Reduced,
    Epochs,
    Barriers,
    Status,
    Scratch,
    Peers,
    Ready,
    GRID: gl.constexpr,
    BAR_STRIDE: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
):
    BPR: gl.constexpr = GRID // 4
    rank = task_index // BPR
    local = task_index % BPR
    r = rank.to(gl.int64)
    tag = _advance_tag(Epochs, task_index)
    tasks = gl.load(NumPost + rank) // 16 * 64
    delayed_point(Seed, task_index, 0)
    if task_index != SKIP:
        for task in range(local, tasks, BPR):
            block = task // 64
            expert = gl.load(Experts + rank * 64 + block)
            expert_tile(
                X,
                Sorted + rank * 1024,
                W13 + r * 64 * 8192 * 2048,
                S13 + r * 64 * 8192 * 128,
                Inter + rank * 64 * 4096,
                block,
                expert,
                task % 64,
                True,
                True,
            )
    delayed_point(Seed, task_index, 1)
    ok = gl.full((), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    if task_index != SKIP:
        ok, _, _ = scoped.counter_barrier(
            Barriers, tag, 10_000_000, GRID, 12, MAX_SPINS=1_048_576
        )
    if ok:
        for task in range(local, tasks, BPR):
            block = task // 64
            expert = gl.load(Experts + rank * 64 + block)
            expert_tile(
                Inter + rank * 64 * 4096,
                Sorted + rank * 1024,
                W2 + r * 64 * 4096 * 2048,
                S2 + r * 64 * 4096 * 128,
                Y + rank * 64 * 4096,
                block,
                expert,
                task % 64,
                False,
                True,
            )
        delayed_point(Seed, task_index, 2)
        ok, _, _ = scoped.counter_barrier(
            Barriers + BAR_STRIDE, tag, 10_000_000, GRID, 12, MAX_SPINS=1_048_576
        )
    if ok:
        for hit in range(local, 64, BPR):
            expert = gl.load(Ids + hit)
            if (expert >= rank * 64) & (expert < (rank + 1) * 64):
                _norm(Y, Gamma, Scaled, Inv, rank * 64 + hit)
        ok, _, _ = scoped.counter_barrier(
            Barriers + 2 * BAR_STRIDE, tag, 10_000_000, GRID, 12, MAX_SPINS=1_048_576
        )
    if ok:
        if local < 8:
            _sum(Ids, Scores, Scaled, Inv, Weighted, Partial, rank, local)
        delayed_point(Seed, task_index, 3)
        ok, _, _ = scoped.counter_barrier(
            Barriers + 3 * BAR_STRIDE, tag, 10_000_000, GRID, 12, MAX_SPINS=1_048_576
        )
    if ok:
        oneshot(
            Partial.to(gl.pointer_type(gl.int32)),
            Scratch,
            Peers,
            None,
            Reduced,
            Ready,
            None,
            4096,
            4,
            BPR,
            128,
            PAYLOAD,
            FLAGS,
            PROFILE=0,
            TAG=tag,
            SCOPED_INPUT=True,
            SCOPED_OUTPUT=True,
            READY_STRIDE=32,
            TASK_INDEX=task_index,
        )
        scoped.drain_stores()
        ok = (
            scoped.load_flag(Ready + (((tag - 1) % 1000) * GRID + task_index) * 32) == 1
        )
    gl.store(Status + task_index, ok)


@g.jit
def _launch(
    X,
    Ids,
    Scores,
    Sorted,
    Experts,
    NumPost,
    W13,
    S13,
    W2,
    S2,
    Gamma,
    Inter,
    Y,
    Scaled,
    Inv,
    Weighted,
    Partial,
    Reduced,
    Epochs,
    Barriers,
    Status,
    Scratch,
    Peers,
    Ready,
    GRID: gl.constexpr,
    BAR_STRIDE: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
):
    stage(
        gl.program_id(0),
        X,
        Ids,
        Scores,
        Sorted,
        Experts,
        NumPost,
        W13,
        S13,
        W2,
        S2,
        Gamma,
        Inter,
        Y,
        Scaled,
        Inv,
        Weighted,
        Partial,
        Reduced,
        Epochs,
        Barriers,
        Status,
        Scratch,
        Peers,
        Ready,
        GRID,
        BAR_STRIDE,
        PAYLOAD,
        FLAGS,
        SKIP,
        Seed,
    )


class Workspace(msgspec.Struct):
    inter: torch.Tensor
    output: torch.Tensor
    scaled: torch.Tensor
    inv: torch.Tensor
    weighted: torch.Tensor
    partial: torch.Tensor
    reduced: torch.Tensor
    epochs: torch.Tensor
    barriers: torch.Tensor
    status: torch.Tensor
    scratch: torch.Tensor
    peers: torch.Tensor
    ready: torch.Tensor
    grid: int
    barrier_stride: int
    payload: tuple[int, int, int]
    flags: tuple[int, int, int]


def make_workspace(device, *, k=9):
    grid = torch.cuda.get_device_properties(device).multi_processor_count * k
    assert grid % 12 == 0 and grid % 4 == 0
    plan = oneshot_plan(4096, grid // 4, 4)

    def empty(shape, dtype=torch.float32):
        return torch.empty(shape, device=device, dtype=dtype)

    def zero(shape):
        return torch.zeros(shape, device=device, dtype=torch.int32)

    scratch = zero(plan.total_words)
    regions = [plan.region(n) for n in ("payload", "flags")]
    stride = (grid // 12 + 1 + grid) * 32
    return Workspace(
        empty((4, 64, 4096), torch.bfloat16),
        empty((4, 64, 4096), torch.bfloat16),
        empty((4, 64, 4096)),
        empty((4, 64)),
        empty((4, 8, 4096)),
        empty((4, 8, 4096), torch.bfloat16),
        empty((4, 8, 4096)),
        zero((grid, 32)),
        zero(4 * stride),
        empty(grid, torch.int32),
        scratch,
        torch.full((4,), scratch.data_ptr(), device=device, dtype=torch.int64),
        zero((1000, grid, 32)),
        grid,
        stride,
        *((r.base, r.rank_stride, r.parity_stride) for r in regions),
    )


def s5(
    x,
    ids,
    scores,
    table,
    weights,
    gamma,
    ws,
    *,
    skip=-1,
    delay_seed=None,
    compile_only=False,
):
    """GPU0 virtual EP4 fixture, native packed weight tensors [4,64,...], serial stream."""
    args = (
        x,
        ids,
        scores,
        *table,
        weights["w13"],
        weights["w13_scale"],
        weights["w2"],
        weights["w2_scale"],
        gamma,
        ws.inter,
        ws.output,
        ws.scaled,
        ws.inv,
        ws.weighted,
        ws.partial,
        ws.reduced,
        ws.epochs,
        ws.barriers,
        ws.status,
        ws.scratch,
        ws.peers,
        ws.ready,
    )
    kwargs = dict(
        GRID=ws.grid,
        BAR_STRIDE=ws.barrier_stride,
        PAYLOAD=ws.payload,
        FLAGS=ws.flags,
        SKIP=skip,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if compile_only:
        return _launch.warmup(*args, **kwargs, grid=(ws.grid,))
    return _launch[(ws.grid,)](*args, **kwargs)

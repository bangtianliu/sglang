"""Common-shape S4 streamed router/pooler, unchanged tie rules, one-wave slot table."""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    chain_guard,
    scoped,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_state import (
    State,
    make_state,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.route import (
    _finish_row,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.slot_table import table
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.streamed_gemm import (
    project_last,
)


@g.jit
def stage(
    pid,
    X,
    W,
    Bias,
    Norm,
    KPart,
    P,
    Epochs,
    Barriers,
    Status,
    Scores,
    Ids,
    Y,
    Sorted,
    Experts,
    NumPost,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    LOCAL_START: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
):
    tag = _advance_tag(Epochs, pid)
    delayed_point(Seed, pid, 0)
    if pid == 0:
        i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [4], [0]))
        scoped.store_words(Sorted + i, gl.full_like(i, 64), i >= 0)
        scoped.store_words(Experts + gl.minimum(i, 63), gl.full_like(i, -1), i < 64)
    ok = project_last(
        pid,
        X,
        W,
        KPart,
        P,
        Barriers,
        Barriers + STRIDE,
        tag,
        4608,
        8192,
        8,
        64,
        GRID,
        False,
        SKIP,
    )
    delayed_point(Seed, pid, 1)
    if ok and pid < 8:
        _finish_row(pid, P, Bias, Norm, Scores, Ids, Y)
        scoped.drain_stores()
        ticket = gl.atomic_add(Barriers + 2 * STRIDE, 1, sem="relaxed", scope="gpu")
        if ticket + 1 == tag * 8:
            scoped.publish(Barriers + 2 * STRIDE + 32, tag)
    if ok and pid == 0:
        seen, _ = scoped.wait_flag(
            Barriers + 2 * STRIDE + 32, tag, 10_000_000, MAX_SPINS=1_048_576
        )
        ok = seen == tag
    delayed_point(Seed, pid, 2)
    if ok and pid == 0:
        table(Ids, Sorted, Experts, NumPost, LOCAL_START)
    gl.store(Status + pid, ok)
    return ok


@g.jit
def _launch(
    X,
    W,
    Bias,
    Norm,
    KPart,
    P,
    Epochs,
    Barriers,
    Status,
    Scores,
    Ids,
    Y,
    Sorted,
    Experts,
    NumPost,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    LOCAL_START: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Failure=None,
    GUARD_MODE: gl.constexpr = 0,
):
    pid = gl.program_id(0)
    if GUARD_MODE == -1:
        mode = gl.load(Failure + 2)
    else:
        mode = gl.full((), GUARD_MODE, gl.int32, gl.BlockedLayout([1], [32], [4], [0]))
    if chain_guard.enabled_runtime(Failure, 4, mode):
        success = stage(
            gl.program_id(0),
            X,
            W,
            Bias,
            Norm,
            KPart,
            P,
            Epochs,
            Barriers,
            Status,
            Scores,
            Ids,
            Y,
            Sorted,
            Experts,
            NumPost,
            GRID,
            STRIDE,
            LOCAL_START,
            SKIP,
            Seed,
        )
        if mode == 0:
            chain_guard.finish(Failure, Status, pid, 4)
        elif mode == 1:
            chain_guard.finish_value(Failure, success, pid, 4)
    else:
        gl.store(Status + pid, 0)


class Workspace(msgspec.Struct):
    projection: torch.Tensor
    partial: torch.Tensor
    state: State
    ids: torch.Tensor
    scores: torch.Tensor
    pooled: torch.Tensor
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_post: torch.Tensor


def make_workspace(device, *, shape=None):
    def empty(shape, dtype):
        return torch.empty(shape, device=device, dtype=dtype)

    return Workspace(
        empty((8, 4608), torch.int32),
        empty((8, 8, 4608), torch.int32),
        make_state(device, phases=3, shape=shape),
        empty((8, 8), torch.int32),
        empty((8, 8), torch.float32),
        empty((8, 4096), torch.bfloat16),
        empty(1024, torch.int32),
        empty(64, torch.int32),
        empty(1, torch.int32),
    )


def routing(
    x,
    weight,
    bias,
    norm,
    *,
    rank=0,
    workspace=None,
    delay_seed=None,
    skip=-1,
    failure=None,
    guard_mode=0,
):
    ws = make_workspace(x.device) if workspace is None else workspace
    s = ws.state
    args = (
        x,
        weight,
        bias,
        norm,
        ws.partial,
        ws.projection,
        s.epochs,
        s.barriers,
        s.status,
        ws.scores,
        ws.ids,
        ws.pooled,
        ws.sorted_ids,
        ws.expert_ids,
        ws.num_post,
    )
    opts = dict(
        GRID=s.grid,
        STRIDE=s.stride,
        LOCAL_START=rank * 64,
        SKIP=skip,
        Seed=delay_seed,
        **s.shape.options(),
        Failure=failure,
        GUARD_MODE=guard_mode,
    )
    key = (
        rank,
        norm.dtype,
        delay_seed is not None,
        skip,
        failure is not None,
        guard_mode,
    )
    if key not in s.admitted:
        kernel = _launch.warmup(*args, **opts, grid=(s.grid,))
        r = s.shape.admit("common-S4", kernel, grid=s.grid)
        s.admitted[key] = dict(
            vgprs=kernel.n_regs,
            lds=kernel.metadata.shared,
            spills=kernel.n_spills,
            occupancy=r.active_ctas_per_cu,
        )
    _launch[s.grid,](*args, **opts)
    return (ws.scores, ws.ids, ws.pooled, ws.sorted_ids, ws.expert_ids, ws.num_post)

"""Common four-wave S6 streamed unpool and unchanged FFN boundary."""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import chain_guard
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_state import (
    State,
    make_state,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.timeline import (
    stamp,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
    norm_residual,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.streamed_gemm import (
    project_last,
)


@g.jit
def stage(
    pid,
    X,
    W,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    KPart,
    P,
    Epochs,
    Barriers,
    Status,
    Out,
    Updated,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    PRE_BF16: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
):
    stamp(Trace, pid, 0)
    tag = _advance_tag(Epochs, pid)
    delayed_point(Seed, pid, 0)
    ok = project_last(
        pid,
        X,
        W,
        KPart,
        P,
        Barriers,
        Barriers + STRIDE,
        tag,
        8192,
        4096,
        4,
        128,
        GRID,
        True,
        SKIP,
        Trace=Trace,
    )
    stamp(Trace, pid, 1)
    delayed_point(Seed, pid, 1)
    if ok and pid < 8:
        norm_residual(
            pid, P, Stream, Post, Pre, Alpha, Beta, Out, Updated, True, PRE_BF16, True
        )
    stamp(Trace, pid, 31)
    gl.store(Status + pid, ok)
    return ok


@g.jit
def _launch(
    X,
    W,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    KPart,
    P,
    Epochs,
    Barriers,
    Status,
    Out,
    Updated,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    PRE_BF16: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
    Failure=None,
    GUARD_MODE: gl.constexpr = 0,
):
    pid = gl.program_id(0)
    if GUARD_MODE == -1:
        mode = gl.load(Failure + 2)
    else:
        mode = gl.full((), GUARD_MODE, gl.int32, gl.BlockedLayout([1], [32], [4], [0]))
    if chain_guard.enabled_runtime(Failure, 6, mode):
        success = stage(
            gl.program_id(0),
            X,
            W,
            Stream,
            Post,
            Pre,
            Alpha,
            Beta,
            KPart,
            P,
            Epochs,
            Barriers,
            Status,
            Out,
            Updated,
            GRID,
            STRIDE,
            PRE_BF16,
            SKIP,
            Seed,
            Trace,
        )
        if mode == 0:
            chain_guard.finish(Failure, Status, pid, 6)
        elif mode == 1:
            chain_guard.finish_value(Failure, success, pid, 6)
    else:
        gl.store(Status + pid, 0)


class Workspace(msgspec.Struct):
    projection: torch.Tensor
    partial: torch.Tensor
    state: State


def make_workspace(device, *, shape=None):
    return Workspace(
        torch.empty((8, 8192), device=device, dtype=torch.int32),
        torch.empty((4, 8, 8192), device=device, dtype=torch.int32),
        make_state(device, phases=2, shape=shape),
    )


def unpool_boundary(
    x,
    weight,
    stream,
    post,
    pre,
    alpha,
    beta,
    *,
    pre_bf16=False,
    workspace=None,
    delay_seed=None,
    skip=-1,
    output=None,
    trace=None,
    failure=None,
    guard_mode=0,
):
    ws = make_workspace(x.device) if workspace is None else workspace
    s = ws.state
    out, updated = (
        (torch.empty_like(stream, dtype=torch.bfloat16), torch.empty_like(stream))
        if output is None
        else output
    )
    args = (
        x,
        weight,
        stream,
        post,
        pre,
        alpha,
        beta,
        ws.partial,
        ws.projection,
        s.epochs,
        s.barriers,
        s.status,
        out,
        updated,
    )
    opts = dict(
        GRID=s.grid,
        STRIDE=s.stride,
        PRE_BF16=pre_bf16,
        SKIP=skip,
        Seed=delay_seed,
        Trace=trace,
        **s.shape.options(),
        Failure=failure,
        GUARD_MODE=guard_mode,
    )
    key = (
        trace is not None,
        pre is None,
        pre_bf16,
        delay_seed is not None,
        skip,
        failure is not None,
        guard_mode,
    )
    if key not in s.admitted:
        kernel = _launch.warmup(*args, **opts, grid=(s.grid,))
        r = s.shape.admit("common-S6", kernel, grid=s.grid)
        s.admitted[key] = dict(
            vgprs=kernel.n_regs,
            lds=kernel.metadata.shared,
            spills=kernel.n_spills,
            occupancy=r.active_ctas_per_cu,
        )
    _launch[s.grid,](*args, **opts)
    return (out, updated)

"""One rank's S3 norm/gate, projection, stand-in peer all-reduce and boundary."""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    chain_guard,
    standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_state import (
    State,
    make_state,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    store_bf16_tile,
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
def norm_gate(X, Gate, Y, head):
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [4, 1], [1, 0])
    r = gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    ptr = r[:, None] * 2048 + head * 64 + d[None, :]
    x = gl.load(X + ptr).to(gl.float32)
    gate = gl.load(Gate + ptr).to(gl.float32)
    x *= gl.rsqrt(gl.sum(x * x, 1) / 64 + 1e-5)[:, None]
    value = (x / (1.0 + gl.exp(-gate))).to(gl.bfloat16)
    store_bf16_tile(Y + ptr, value, r[:, None] >= 0, 8, 64)


@g.jit
def stage(
    pid,
    X,
    Gate,
    W,
    Stream,
    Pre,
    Alpha,
    Beta,
    Normalized,
    KPart,
    P,
    Partial,
    Epochs,
    Barriers,
    Status,
    Scratch,
    Peers,
    Ready,
    Reduced,
    Out,
    Updated,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    RANK: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
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
        2048,
        4,
        128,
        GRID,
        True,
        SKIP,
        Partial + RANK * 8 * 8192,
        Trace,
        Gate,
    )
    stamp(Trace, pid, 3)
    delayed_point(Seed, pid, 1)
    if ok and pid < 8:
        value, ok = standin.reduce_row(
            Partial,
            Scratch,
            Peers,
            Reduced,
            tag,
            pid,
            8192,
            GRID,
            RANK,
            PAYLOAD,
            FLAGS,
            Trace,
        )
        stamp(Trace, pid, 4)
        delayed_point(Seed, pid, 2)
        if ok:
            norm_residual(
                pid,
                Reduced + RANK * 8 * 8192,
                Stream,
                None,
                Pre,
                Alpha,
                Beta,
                Out,
                Updated,
                False,
                False,
                True,
                Value=value,
            )
    stamp(Trace, pid, 31)
    gl.store(Status + pid, ok)
    return ok


@g.jit
def _launch(
    X,
    Gate,
    W,
    Stream,
    Pre,
    Alpha,
    Beta,
    Normalized,
    KPart,
    P,
    Partial,
    Epochs,
    Barriers,
    Status,
    Scratch,
    Peers,
    Ready,
    Reduced,
    Out,
    Updated,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    RANK: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
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
    if chain_guard.enabled_runtime(Failure, 3, mode):
        success = stage(
            gl.program_id(0),
            X,
            Gate,
            W,
            Stream,
            Pre,
            Alpha,
            Beta,
            Normalized,
            KPart,
            P,
            Partial,
            Epochs,
            Barriers,
            Status,
            Scratch,
            Peers,
            Ready,
            Reduced,
            Out,
            Updated,
            GRID,
            STRIDE,
            RANK,
            PAYLOAD,
            FLAGS,
            SKIP,
            Seed,
            Trace,
        )
        if mode == 0:
            chain_guard.finish(Failure, Status, pid, 3)
        elif mode == 1:
            chain_guard.finish_value(Failure, success, pid, 3)
    else:
        gl.store(Status + pid, 0)


class Workspace(msgspec.Struct):
    normalized: torch.Tensor
    k_partial: torch.Tensor
    projection: torch.Tensor
    partial: torch.Tensor
    state: State
    peers: standin.Standins


def make_workspace(device, *, shape=None):
    state = make_state(device, phases=4, shape=shape)
    return Workspace(
        torch.empty((8, 2048), device=device, dtype=torch.bfloat16),
        torch.empty((4, 8, 8192), device=device, dtype=torch.int32),
        torch.empty((8, 8192), device=device, dtype=torch.int32),
        torch.empty((4, 8, 8192), device=device, dtype=torch.bfloat16),
        state,
        standin.make_standins(device, width=8192, ctas=state.grid),
    )


def attention_tail(
    x,
    gate,
    weight,
    stream,
    pre,
    alpha,
    beta,
    *,
    rank,
    workspace,
    delay_seed=None,
    skip=-1,
    output=None,
    trace=None,
    failure=None,
    guard_mode=0,
):
    ws = workspace
    s = ws.state
    p = ws.peers
    out, updated = (
        (torch.empty_like(stream, dtype=torch.bfloat16), torch.empty_like(stream))
        if output is None
        else output
    )
    args = (
        x,
        gate,
        weight,
        stream,
        pre,
        alpha,
        beta,
        ws.normalized,
        ws.k_partial,
        ws.projection,
        ws.partial,
        s.epochs,
        s.barriers,
        s.status,
        p.scratch,
        p.peers,
        p.ready,
        p.reduced,
        out,
        updated,
    )
    opts = dict(
        GRID=s.grid,
        STRIDE=s.stride,
        RANK=rank,
        PAYLOAD=p.payload,
        FLAGS=p.flags,
        SKIP=skip,
        Seed=delay_seed,
        Trace=trace,
        **s.shape.options(),
        Failure=failure,
        GUARD_MODE=guard_mode,
    )
    key = (
        trace is not None,
        rank,
        delay_seed is not None,
        skip,
        pre.dtype,
        failure is not None,
        guard_mode,
    )
    if key not in s.admitted:
        kernel = _launch.warmup(*args, **opts, grid=(s.grid,))
        r = s.shape.admit("common-S3", kernel, grid=s.grid)
        s.admitted[key] = dict(
            vgprs=kernel.n_regs,
            lds=kernel.metadata.shared,
            spills=kernel.n_spills,
            occupancy=r.active_ctas_per_cu,
        )
    _launch[s.grid,](*args, **opts)
    return (out, updated)

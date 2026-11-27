"""S1 four-wave streamed projection plus the established QKV/cache epilogue."""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    chain_guard,
    scoped,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
    CommonShape,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.timeline import (
    stamp,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
    head_epilogue,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.streamed_gemm import (
    partial_tile,
    reduce_tile,
)


class Workspace(msgspec.Struct):
    projection: torch.Tensor
    state: torch.Tensor
    partial: torch.Tensor
    barriers: torch.Tensor
    status: torch.Tensor
    ctas: int
    stride: int
    shape: CommonShape
    admitted: dict = msgspec.field(default_factory=dict)


def make_workspace(device, *, ctas=None, shape=None):
    shape = CommonShape() if shape is None else shape
    ctas = shape.grid(device) if ctas is None else ctas
    assert 0 < ctas <= shape.grid(device) and ctas % 12 == 0
    stride = (ctas // 12 + 1 + ctas) * 32
    return Workspace(
        torch.empty((8, 4608), device=device, dtype=torch.int32),
        torch.zeros(ctas * 32 + 72, device=device, dtype=torch.int32),
        torch.empty((8, 8, 4608), device=device, dtype=torch.int32),
        torch.zeros(2 * stride, device=device, dtype=torch.int32),
        torch.empty(ctas, device=device, dtype=torch.int32),
        ctas,
        stride,
        shape,
    )


@g.jit
def stage(
    pid,
    X,
    W,
    P,
    State,
    Partial,
    Barriers,
    Status,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    Q,
    K,
    V,
    G,
    KS,
    VS,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
):
    stamp(Trace, pid, 0)
    tag = _advance_tag(State, pid)
    delayed_point(Seed, pid, 0)
    ok = pid != SKIP
    if pid != SKIP:
        for task in range(pid, 72 * 8, GRID):
            tile = task // 8
            partial_tile(X, W, Partial, task, 4608, 8192, 8, 64, Trace=Trace)
            stamp(Trace, pid, 1)
            delayed_point(Seed, pid, 1)
            scoped.drain_stores()
            ticket = gl.atomic_add(Barriers + tile * 32, 1, sem="relaxed", scope="gpu")
            stamp(Trace, pid, 2)
            if ticket + 1 == tag * 8:
                value = reduce_tile(Partial, P, tile, 4608, 8, True)
                stamp(Trace, pid, 3)
                delayed_point(Seed, pid, 2)
                head_epilogue(
                    P,
                    tile,
                    POS,
                    COS,
                    LOC,
                    MAP,
                    KC,
                    VC,
                    Q,
                    K,
                    V,
                    G,
                    KS,
                    VS,
                    EPS,
                    SCALE,
                    ROPE,
                    HAS_MAP,
                    value,
                )
                gl.store(State + GRID * 32 + tile, tag)
    stamp(Trace, pid, 31)
    gl.store(Status + pid, ok)
    return ok


@g.jit
def _launch(
    X,
    W,
    P,
    State,
    Partial,
    Barriers,
    Status,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    Q,
    K,
    V,
    G,
    KS,
    VS,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
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
    if chain_guard.enabled_runtime(Failure, 1, mode):
        success = stage(
            gl.program_id(0),
            X,
            W,
            P,
            State,
            Partial,
            Barriers,
            Status,
            POS,
            COS,
            LOC,
            MAP,
            KC,
            VC,
            Q,
            K,
            V,
            G,
            KS,
            VS,
            GRID,
            STRIDE,
            EPS,
            SCALE,
            ROPE,
            HAS_MAP,
            SKIP,
            Seed,
            Trace,
        )
        if mode == 0:
            chain_guard.finish(Failure, Status, pid, 1)
        elif mode == 1:
            chain_guard.finish_value(Failure, success, pid, 1)
    else:
        gl.store(Status + pid, 0)


def qkvg(
    x,
    weight,
    positions,
    cos_sin,
    cache_k,
    cache_v,
    locations,
    mapping,
    eps,
    q_scale,
    k_scale,
    v_scale,
    *,
    ctas=None,
    workspace=None,
    shape=None,
    skip=-1,
    delay_seed=None,
    trace=None,
    failure=None,
    guard_mode=0,
):
    ws = (
        make_workspace(x.device, ctas=ctas, shape=shape)
        if workspace is None
        else workspace
    )
    assert ctas is None or ctas == ws.ctas
    q = torch.empty((8, 2048), device=x.device, dtype=torch.bfloat16)
    k = torch.empty((8, 256), device=x.device, dtype=torch.bfloat16)
    v, g = (torch.empty_like(k), torch.empty_like(q))
    args = (
        x,
        weight,
        ws.projection,
        ws.state,
        ws.partial,
        ws.barriers,
        ws.status,
        positions,
        cos_sin,
        locations,
        mapping,
        cache_k,
        cache_v,
        q,
        k,
        v,
        g,
        k_scale,
        v_scale,
    )
    opts = dict(
        GRID=ws.ctas,
        STRIDE=ws.stride,
        EPS=float(eps),
        SCALE=float(q_scale),
        ROPE=cos_sin is not None,
        HAS_MAP=mapping is not None,
        SKIP=skip,
        Seed=delay_seed,
        Trace=trace,
        num_warps=ws.shape.waves,
        waves_per_eu=ws.shape.k,
        enable_fp_fusion=False,
        Failure=failure,
        GUARD_MODE=guard_mode,
    )
    key = (
        trace is not None,
        cos_sin is not None,
        mapping is not None,
        skip,
        delay_seed is not None,
        float(eps),
        float(q_scale),
        failure is not None,
        guard_mode,
    )
    if key not in ws.admitted:
        kernel = _launch.warmup(*args, **opts, grid=(ws.ctas,))
        resource = ws.shape.admit("common-S1", kernel, grid=ws.ctas)
        ws.admitted[key] = dict(
            vgprs=kernel.n_regs,
            lds=kernel.metadata.shared,
            spills=kernel.n_spills,
            occupancy=resource.active_ctas_per_cu,
        )
    _launch[ws.ctas,](*args, **opts)
    return (q, k, v, g)

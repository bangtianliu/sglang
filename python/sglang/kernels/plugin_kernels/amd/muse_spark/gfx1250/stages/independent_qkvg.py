"""S1 trial: two-wave TDM SK8 projection and unchanged QKV/cache epilogue."""

import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import chain_guard
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_qkvg,
    wide_projection,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
    head_epilogue,
)

_make_common_workspace = common_qkvg.make_workspace


@g.jit
def qkvg_heads(
    P,
    State,
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
    Failure,
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
    Seed=None,
):
    head = gl.program_id(0)
    tag = _advance_tag(State, head)
    j = gl.arange(0, 4, layout=gl.BlockedLayout([1], [32], [4], [0]))
    ok = gl.min(gl.load(Status + head * 4 + j), 0) == 1
    delayed_point(Seed, head, 1)
    if ok:
        # Match reduce_tile's result layout; retain all existing epilogue math.
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
        row = gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
        col = head * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
        value = gl.load(P + row[:, None] * 4608 + col[None, :]).to(
            gl.float32, bitcast=True
        )
        head_epilogue(
            P,
            head,
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
        gl.store(State + 288 * 32 + head, tag)
    chain_guard.finish_value(Failure, ok, head, 1)


def make_workspace(device, *, ctas=None, shape=None):
    assert ctas is None or ctas == 288
    return _make_common_workspace(device, ctas=288, shape=shape)


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
    assert guard_mode == 0 and trace is None
    ws = (
        make_workspace(x.device, ctas=ctas, shape=shape)
        if workspace is None
        else workspace
    )
    assert ws.ctas == 288 and (ctas is None or ctas == 288)
    q = torch.empty((8, 2048), device=x.device, dtype=torch.bfloat16)
    k = torch.empty((8, 256), device=x.device, dtype=torch.bfloat16)
    v, gout = torch.empty_like(k), torch.empty_like(q)
    pa = (x, weight, ws.projection, ws.status, failure)
    po = dict(
        N=4608,
        K=8192,
        STAGE=1,
        SK=8,
        SKIP=skip,
        Seed=delay_seed,
        num_warps=2,
        enable_fp_fusion=False,
    )
    ha = (
        ws.projection,
        ws.state,
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
        gout,
        k_scale,
        v_scale,
        failure,
    )
    ho = dict(
        EPS=float(eps),
        SCALE=float(q_scale),
        ROPE=cos_sin is not None,
        HAS_MAP=mapping is not None,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
    )
    key = (
        skip,
        delay_seed is not None,
        cos_sin is not None,
        mapping is not None,
        failure is not None,
        float(eps),
        float(q_scale),
    )
    if key not in ws.admitted:
        records = []
        for name, fn, aa, kk, grid in [
            ("projection", wide_projection.project, pa, po, 288),
            ("heads", qkvg_heads, ha, ho, 72),
        ]:
            kernel = fn.warmup(*aa, **kk, grid=(grid,))
            r = kernel_occupancy("independent-S1-" + name, kernel)
            assert r.active_ctas_per_cu >= 1 and kernel.metadata.shared <= 163840
            records.append(
                dict(
                    phase=name,
                    waves=kernel.metadata.num_warps,
                    vgprs=kernel.n_regs,
                    spills=kernel.n_spills,
                    lds=kernel.metadata.shared,
                    occupancy=r.active_ctas_per_cu,
                    grid=grid,
                )
            )
        ws.admitted[key] = records
    wide_projection.project[(288,)](*pa, **po)
    qkvg_heads[(72,)](*ha, **ho)
    return q, k, v, gout

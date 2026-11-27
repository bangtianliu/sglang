"""S2 trial: dispatch-separated partial/merge, unchanged split and merge order."""

import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import chain_guard
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attention as original,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    pipelined_attention as pipe,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)


@g.jit
def partials(
    Q,
    K,
    V,
    Pages,
    Map,
    Length,
    KS,
    VS,
    P,
    LSE,
    Epochs,
    Status,
    Failure,
    SPLITS: gl.constexpr,
    MAX_PAGES: gl.constexpr,
    SCALE: gl.constexpr,
    LEFT: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAP_PAGE: gl.constexpr,
    SKIP: gl.constexpr = -1,
    PIPELINED: gl.constexpr = False,
    Seed=None,
):
    pid = gl.program_id(0)
    tag = _advance_tag(Epochs, pid)
    allowed = chain_guard.enabled(Failure, 2)
    ok = allowed & (pid != SKIP)
    delayed_point(Seed, pid, 0)
    if ok:
        split = pid // 4
        head = pid % 4
        if PIPELINED:
            value, lse = pipe.partial(
                Q,
                K,
                V,
                Pages,
                Map,
                Length,
                KS,
                VS,
                head,
                split,
                SCALE,
                LEFT,
                SPLITS,
                HAS_MAP,
                MAP_PAGE,
                MAX_PAGES,
            )
        else:
            value, lse = original.partial(
                Q,
                K,
                V,
                Pages,
                Map,
                Length,
                KS,
                VS,
                head,
                split,
                SCALE,
                LEFT,
                SPLITS,
                HAS_MAP,
                MAP_PAGE,
                MAX_PAGES,
            )
        layout: gl.constexpr = value.type.layout
        row = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
        col = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
        gl.store(
            P + ((head * 64 + row[:, None]) * SPLITS + split) * 64 + col[None, :],
            value.to(gl.int32, bitcast=True),
        )
        gl.store(
            LSE + (head * 64 + row) * SPLITS + split, lse.to(gl.int32, bitcast=True)
        )
    delayed_point(Seed, pid, 1)
    gl.store(Status + pid, ok)
    if allowed:
        chain_guard.finish_value(Failure, ok, pid, 2)


@g.jit
def merge(P, LSE, Status, CtaStatus, Out, Failure, SPLITS: gl.constexpr, Seed=None):
    pid = gl.program_id(0)
    ok = gl.load(Failure) == 0
    delayed_point(Seed, pid, 2)
    if ok:
        original._merge(
            P, LSE, None, Status, Out, pid, 1, SPLITS, None, pid, ASSUME_READY=True
        )
    else:
        gl.store(Status + pid, 0)
        if pid < SPLITS * 4:
            gl.store(CtaStatus + pid, 0)
        if pid + 256 < SPLITS * 4:
            gl.store(CtaStatus + pid + 256, 0)


class Workspace(original.Workspace):
    local_failure: object = None


def make_workspace(device, *, context, window_left=-1, **kwargs):
    b = original.make_workspace(device, context=context, window_left=window_left)
    b.ctas = b.splits * 4
    return Workspace(
        b.partial,
        b.lse,
        torch.zeros((b.ctas, 32), device=device, dtype=torch.int32),
        b.counters,
        b.flags,
        b.status,
        torch.empty((b.ctas,), device=device, dtype=torch.int32),
        b.splits,
        b.ctas,
        b.pages_per_chunk,
        b.shape,
        {},
        torch.zeros((2,), device=device, dtype=torch.int32),
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
    window_left=-1,
    mapping=None,
    mapping_page_aligned=False,
    workspace=None,
    ctas=None,
    block_n=64,
    waves=4,
    softmax_scale=0.125,
    skip=-1,
    delay_seed=None,
    shape=None,
    output=None,
    failure=None,
    guard_mode=0,
):
    assert guard_mode == 0 and block_n == 64 and waves == 4
    ws = (
        make_workspace(q.device, context=context, window_left=window_left)
        if workspace is None
        else workspace
    )
    failure = ws.local_failure if failure is None else failure
    output = torch.empty_like(q) if output is None else output
    pa = (
        q,
        k,
        v,
        page_table,
        mapping,
        length,
        k_descale,
        v_descale,
        ws.partial,
        ws.lse,
        ws.epochs,
        ws.cta_status,
        failure,
    )
    po = dict(
        SPLITS=ws.splits,
        MAX_PAGES=ws.pages_per_chunk,
        SCALE=softmax_scale,
        LEFT=window_left,
        HAS_MAP=mapping is not None,
        MAP_PAGE=mapping_page_aligned,
        SKIP=skip,
        PIPELINED=ws.pages_per_chunk > 8,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
    )
    ma = (ws.partial, ws.lse, ws.status, ws.cta_status, output, failure)
    mo = dict(SPLITS=ws.splits, Seed=delay_seed, num_warps=4, enable_fp_fusion=False)
    key = (
        window_left,
        mapping is not None,
        mapping_page_aligned,
        skip,
        delay_seed is not None,
        softmax_scale,
    )
    if key not in ws.admitted:
        records = []
        for name, fn, aa, kk, grid in [
            ("partial", partials, pa, po, ws.ctas),
            ("merge", merge, ma, mo, 256),
        ]:
            compiled = fn.warmup(*aa, **kk, grid=(grid,))
            r = kernel_occupancy("split-S2-" + name, compiled)
            assert r.active_ctas_per_cu >= 1
            records.append(
                dict(
                    phase=name,
                    waves=compiled.metadata.num_warps,
                    vgprs=compiled.n_regs,
                    spills=compiled.n_spills,
                    lds=compiled.metadata.shared,
                    occupancy=r.active_ctas_per_cu,
                    grid=grid,
                )
            )
        ws.admitted[key] = records
    partials[(ws.ctas,)](*pa, **po)
    merge[(256,)](*ma, **mo)
    return output

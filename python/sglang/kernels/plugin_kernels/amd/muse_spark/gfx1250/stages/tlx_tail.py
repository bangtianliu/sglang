"""T1 S6 in tl/uTLX; independent budgets, legacy fixed tree and boundary math."""

import msgspec
import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard,
    tlx_workspace,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
    require_pin_support,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_projection,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_boundary import (
    norm_residual,
)


@triton.jit
def finish(
    P,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    Status,
    Failure,
    PRE_BF16: tl.constexpr,
    PIN_LAYOUT: tl.constexpr = True,
    PACKED_INPUT: tl.constexpr = False,
):
    row = tl.program_id(0)
    i = tlx.require_layout(tl.arange(0, 512), ROW1)
    ok = tl.min(tl.load(Status + i), 0) == 1
    if ok:
        norm_residual(
            row,
            P,
            Stream,
            Post,
            Pre,
            Alpha,
            Beta,
            Out,
            Updated,
            True,
            PRE_BF16,
            PIN_LAYOUT=PIN_LAYOUT,
            BRANCH_BF16=PACKED_INPUT,
        )
    else:
        tl.store(Status + row, 0)
    tlx_guard.finish(Failure, ok, row, 6)


class Workspace(msgspec.Struct):
    projection: object
    partial: object
    state: object
    kernels: dict = msgspec.field(default_factory=dict)
    packed: object = None


def make_workspace(device):
    b = tlx_workspace.make_s6(device)
    b.state.status = torch.empty((512,), device=device, dtype=torch.int32)
    return Workspace(
        b.projection,
        b.partial,
        b.state,
        packed=torch.empty((8, 8192), device=device, dtype=torch.bfloat16),
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
    workspace,
    pre_bf16=False,
    skip=-1,
    delay_seed=None,
    output=None,
    trace=None,
    failure=None,
    guard_mode=0,
    pin_layout=True,
    packed_handoff=False,
    projection_depth=4,
):
    assert guard_mode == 0 and trace is None
    if pin_layout:
        require_pin_support()
    ws = workspace
    s = ws.state
    out, updated = (
        (torch.empty_like(stream, dtype=torch.bfloat16), torch.empty_like(stream))
        if output is None
        else output
    )
    pa = (x, weight, ws.projection, s.status)
    po = dict(
        N=8192,
        K=4096,
        STAGE=6,
        DEPTH=projection_depth,
        Packed=ws.packed if packed_handoff else None,
        Failure=failure,
        SKIP=skip,
        Seed=delay_seed,
        num_warps=2,
        num_stages=1,
        enable_fp_fusion=False,
    )
    fa = (
        ws.packed if packed_handoff else ws.projection,
        stream,
        post,
        pre,
        alpha,
        beta,
        out,
        updated,
        s.status,
        failure,
    )
    fo = dict(
        PRE_BF16=pre_bf16,
        PIN_LAYOUT=pin_layout,
        PACKED_INPUT=packed_handoff,
        num_warps=4,
        enable_fp_fusion=False,
    )
    key = (
        skip,
        delay_seed is not None,
        pre is None,
        pre_bf16,
        failure is not None,
        pre.dtype if pre is not None else None,
        post.dtype,
        pin_layout,
        packed_handoff,
        projection_depth,
    )
    if key not in s.admitted:
        records = []
        kernels = []
        for name, fn, args, opts, grid in [
            ("projection", tlx_projection.project, pa, po, 512),
            ("boundary", finish, fa, fo, 8),
        ]:
            k = fn.warmup(*args, **opts, grid=(grid,))
            r = kernel_occupancy("tlx-S6-" + name, k)
            assert r.active_ctas_per_cu >= 1 and k.metadata.shared <= 163840
            records.append(
                dict(
                    phase=name,
                    vgprs=k.n_regs,
                    spills=k.n_spills,
                    lds=k.metadata.shared,
                    occupancy=r.active_ctas_per_cu,
                    waves=k.metadata.num_warps,
                    grid=grid,
                )
            )
            kernels.append(k)
        s.admitted[key] = records
        ws.kernels[key] = kernels
    tlx_projection.project[(8192 // tlx_projection.BN_S6,)](
        *pa, **{**po, **({"num_warps": 4} if tlx_projection.BN_S6 == 64 else {})},
        EARLY=tlx_projection.EARLY_ISSUE, COPY=tlx_projection.COPY_MODE,
        BN=tlx_projection.BN_S6)
    finish[(8,)](*fa, **fo)
    return out, updated

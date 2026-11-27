"""S6 trial: two-wave large-LDS projection, separate unchanged row boundary."""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import chain_guard
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_tail,
    wide_projection,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    norm_residual,
)

_make_common_workspace = common_tail.make_workspace


@g.jit
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
    PRE_BF16: gl.constexpr,
):
    row = gl.program_id(0)
    i = gl.arange(0, 512, layout=gl.BlockedLayout([1], [32], [4], [0]))
    # One-shot dispatch-boundary status reads, never a shared poll loop.
    ok = gl.min(gl.load(Status + i), 0) == 1
    if ok:
        norm_residual(
            row, P, Stream, Post, Pre, Alpha, Beta, Out, Updated, True, PRE_BF16, True
        )
    else:
        gl.store(Status + row, 0)
    chain_guard.finish_value(Failure, ok, row, 6)


class Workspace(msgspec.Struct):
    projection: torch.Tensor
    partial: torch.Tensor
    state: object


def make_workspace(device):
    ws = _make_common_workspace(device)
    ws.state.status = torch.empty((512,), device=device, dtype=torch.int32)
    return Workspace(ws.projection, ws.partial, ws.state)


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
):
    assert guard_mode == 0 and trace is None
    ws = workspace
    s = ws.state
    out, updated = (
        (torch.empty_like(stream, dtype=torch.bfloat16), torch.empty_like(stream))
        if output is None
        else output
    )
    args = (x, weight, ws.projection, s.status, failure)
    opts = dict(
        N=8192,
        K=4096,
        STAGE=6,
        SKIP=skip,
        Seed=delay_seed,
        num_warps=2,
        enable_fp_fusion=False,
    )
    fargs = (
        ws.projection,
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
    fopts = dict(PRE_BF16=pre_bf16, num_warps=4, enable_fp_fusion=False)
    key = (skip, delay_seed is not None, pre is None, pre_bf16, failure is not None)
    if key not in s.admitted:
        records = []
        for name, fn, aa, kk, grid in [
            ("projection", wide_projection.project, args, opts, 512),
            ("boundary", finish, fargs, fopts, 8),
        ]:
            k = fn.warmup(*aa, **kk, grid=(grid,))
            r = kernel_occupancy("independent-S6-" + name, k)
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
        s.admitted[key] = records
    wide_projection.project[(512,)](*args, **opts)
    finish[(8,)](*fargs, **fopts)
    return out, updated

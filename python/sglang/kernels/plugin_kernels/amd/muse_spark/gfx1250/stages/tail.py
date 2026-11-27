"""S6 BF16 unpool and FFN boundary on the shared M8 core in one launch.

Input is the BF16 all-reduced expert sum. Preserve GEMM BF16 rounding, then
BF16 input scaling before post-RMSNorm, BF16 norm output, gated FP32 stream,
and optional BF16 pre-norm input for the last layer. No expert/collective work
is performed here. Reuse workspace serially; discard it after a false status.
"""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    StageState,
    delayed_point,
    handoff,
    make_state,
    norm_residual,
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


class S6Workspace(msgspec.Struct):
    projection: torch.Tensor
    state: StageState


def make_workspace(device):
    return S6Workspace(
        torch.empty((8, 8192), device=device, dtype=torch.int32),
        make_state(device, ctas=192),
    )


@g.jit
def s6_stage(
    task_index,
    ctas,
    X,
    W,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    P,
    Epochs,
    Arrivals,
    Flags,
    Status,
    Out,
    Updated,
    PRE_BF16: gl.constexpr,
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
        if TRACE is not None:
            gl.store(TRACE + task_index * 16 + 12, sync.clock())
        if task_index < 128:
            cached = load_input(X, 4096, 4, 128)
            value = skinny_tile(
                W, cached, task_index, 8192, 4096, 64, 128, 4, 8, 2, ".cs", False
            )
            store_projection(P, task_index, value, 8192, 64, True)
        if TRACE is not None:
            gl.store(TRACE + task_index * 16 + 1, sync.clock())
        delayed_point(Seed, task_index, 1)
        ready = handoff(Arrivals, Flags, tag, 0, task_index, 192)
        if TRACE is not None:
            gl.store(TRACE + task_index * 16 + 2, sync.clock())
        delayed_point(Seed, task_index, 2)
        if ready and task_index < 8:
            norm_residual(
                task_index,
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
                True,
                TRACE + task_index * 16 if TRACE is not None else None,
            )
        gl.store(Status + task_index, ready.to(gl.int32))
        if TRACE is not None:
            gl.store(TRACE + task_index * 16 + 11, sync.clock())


@g.jit
def _standalone(
    X,
    W,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    P,
    Epochs,
    Arrivals,
    Flags,
    Status,
    Out,
    Updated,
    PRE_BF16: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    TRACE=None,
):
    s6_stage(
        gl.program_id(0),
        192,
        X,
        W,
        Stream,
        Post,
        Pre,
        Alpha,
        Beta,
        P,
        Epochs,
        Arrivals,
        Flags,
        Status,
        Out,
        Updated,
        PRE_BF16,
        SKIP,
        Seed,
        TRACE,
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
):
    assert x.shape == (8, 4096) and weight.shape == (8192, 4096)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert stream.shape == (8, 8192) and stream.dtype == torch.float32
    assert post.shape == alpha.shape == beta.shape == (8192,)
    assert post.dtype == torch.bfloat16 and alpha.dtype == beta.dtype == torch.float32
    tensors = [x, weight, stream, post, alpha, beta]
    if pre is not None:
        assert pre.shape == (8192,) and pre.dtype == torch.bfloat16
        tensors.append(pre)
    assert all(t.device == x.device and t.is_contiguous() for t in tensors)
    if workspace is None:
        workspace = make_workspace(x.device)
    s = workspace.state
    validate_state(s, x.device, ctas=192)
    assert (
        workspace.projection.shape == (8, 8192)
        and workspace.projection.dtype == torch.int32
    )
    assert (
        workspace.projection.device == x.device and workspace.projection.is_contiguous()
    )
    out = torch.empty((8, 8192), device=x.device, dtype=torch.bfloat16)
    updated = torch.empty((8, 8192), device=x.device, dtype=torch.float32)
    _standalone[(192,)](
        x,
        weight,
        stream,
        post,
        pre,
        alpha,
        beta,
        workspace.projection,
        s.epochs,
        s.arrivals,
        s.flags,
        s.status,
        out,
        updated,
        pre_bf16,
        Seed=delay_seed,
        num_warps=16,
        enable_fp_fusion=False,
    )
    return out, updated

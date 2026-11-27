"""Language-neutral allocations retained from the frozen Gluon test oracle.

No kernels or Gluon imports. Stage launch code owns its independent geometry.
"""

import msgspec
import torch
import triton

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_peers as standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
    CommonShape,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_state import (
    State,
    make_state,
)


class S1Workspace(msgspec.Struct):
    projection: torch.Tensor
    state: torch.Tensor
    partial: torch.Tensor
    barriers: torch.Tensor
    status: torch.Tensor
    ctas: int
    stride: int
    shape: CommonShape
    admitted: dict = msgspec.field(default_factory=dict)


def make_s1(device, *, ctas=None, shape=None):
    shape = CommonShape() if shape is None else shape
    ctas = shape.grid(device) if ctas is None else ctas
    assert 0 < ctas <= shape.grid(device) and ctas % 12 == 0
    stride = (ctas // 12 + 1 + ctas) * 32
    return S1Workspace(
        torch.empty((8, 4608), device=device, dtype=torch.int32),
        torch.zeros(ctas * 32 + 72, device=device, dtype=torch.int32),
        torch.empty((8, 8, 4608), device=device, dtype=torch.int32),
        torch.zeros(2 * stride, device=device, dtype=torch.int32),
        torch.empty(ctas, device=device, dtype=torch.int32),
        ctas,
        stride,
        shape,
    )


class S2Workspace(msgspec.Struct):
    partial: torch.Tensor
    lse: torch.Tensor
    epochs: torch.Tensor
    counters: torch.Tensor
    flags: torch.Tensor
    status: torch.Tensor
    cta_status: torch.Tensor
    splits: int
    ctas: int
    pages_per_chunk: int
    shape: CommonShape
    admitted: dict = msgspec.field(default_factory=dict)


# Full-context S2 split cap (experiment knob; 128 = the archive-exact layout).
S2_MAX_SPLITS = 128


def make_s2(
    device,
    *,
    context,
    window_left=-1,
    ctas=None,
    block_n=64,
    sliding_splits=32,
    shape=None,
):
    shape = CommonShape() if shape is None else shape
    ctas = shape.grid(device) if ctas is None else ctas
    assert 0 < ctas <= shape.grid(device) and ctas % 12 == 0 and (block_n == 64)
    active = context if window_left < 0 else min(context, window_left + 8)
    tiles = triton.cdiv(active + (63 if window_left >= 0 else 0), 64)
    splits = (
        min(S2_MAX_SPLITS, ctas // 4, max(16, tiles))
        if window_left < 0
        else sliding_splits
    )
    pages = triton.next_power_of_2(triton.cdiv(tiles, splits) * 4)
    counters = torch.zeros(
        (ctas // 12 + 1 + ctas) * 32, device=device, dtype=torch.int32
    )
    flags = counters[(ctas // 12 + 1) * 32 : (ctas // 12 + 1 + 256) * 32].view(256, 32)
    return S2Workspace(
        torch.empty((splits, 256, 64), device=device, dtype=torch.int32),
        torch.empty((splits, 256), device=device, dtype=torch.int32),
        torch.zeros((ctas, 32), device=device, dtype=torch.int32),
        counters,
        flags,
        torch.empty(256, device=device, dtype=torch.int32),
        torch.empty(ctas, device=device, dtype=torch.int32),
        splits,
        ctas,
        pages,
        shape,
    )


class S3Workspace(msgspec.Struct):
    normalized: torch.Tensor
    k_partial: torch.Tensor
    projection: torch.Tensor
    partial: torch.Tensor
    state: State
    peers: standin.Standins


def make_s3(device, *, shape=None):
    state = make_state(device, phases=4, shape=shape)
    return S3Workspace(
        torch.empty((8, 2048), device=device, dtype=torch.bfloat16),
        torch.empty((4, 8, 8192), device=device, dtype=torch.int32),
        torch.empty((8, 8192), device=device, dtype=torch.int32),
        torch.empty((4, 8, 8192), device=device, dtype=torch.bfloat16),
        state,
        standin.make_standins(device, width=8192, ctas=state.grid),
    )


class S4Workspace(msgspec.Struct):
    projection: torch.Tensor
    partial: torch.Tensor
    state: State
    ids: torch.Tensor
    scores: torch.Tensor
    pooled: torch.Tensor
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_post: torch.Tensor


def make_s4(device, *, shape=None):

    def empty(shape, dtype):
        return torch.empty(shape, device=device, dtype=dtype)

    return S4Workspace(
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


class S5Workspace(msgspec.Struct):
    inter: torch.Tensor
    output: torch.Tensor
    scaled: torch.Tensor
    inv: torch.Tensor
    weighted: torch.Tensor
    partial: torch.Tensor
    state: State
    k_partial: torch.Tensor
    peers: standin.Standins


def make_s5(device, *, shape=None):
    s = make_state(device, phases=17, shape=shape)

    def empty(shape, dtype=torch.float32):
        return torch.empty(shape, device=device, dtype=dtype)

    return S5Workspace(
        empty((4, 64, 4096), torch.bfloat16),
        empty((4, 64, 4096), torch.bfloat16),
        empty((4, 64, 4096)),
        empty((4, 64, 32)),
        empty((4, 8, 4096)),
        empty((4, 8, 4096), torch.bfloat16),
        s,
        empty((2, 4, 64, 4096), torch.int32),
        standin.make_standins(device, width=4096, ctas=s.grid),
    )


class S6Workspace(msgspec.Struct):
    projection: torch.Tensor
    partial: torch.Tensor
    state: State


def make_s6(device, *, shape=None):
    return S6Workspace(
        torch.empty((8, 8192), device=device, dtype=torch.int32),
        torch.empty((4, 8, 8192), device=device, dtype=torch.int32),
        make_state(device, phases=2, shape=shape),
    )

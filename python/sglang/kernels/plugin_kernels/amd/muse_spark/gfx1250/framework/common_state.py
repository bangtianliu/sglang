"""Serial launch epochs and bounded two-level barriers for common-shape stages."""

import msgspec
import torch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
    CommonShape,
)


class State(msgspec.Struct):
    epochs: torch.Tensor
    barriers: torch.Tensor
    status: torch.Tensor
    grid: int
    stride: int
    shape: CommonShape
    admitted: dict = msgspec.field(default_factory=dict)


def make_state(device, *, phases, shape=None):
    shape = CommonShape() if shape is None else shape
    grid = shape.grid(device)
    assert grid % 12 == 0
    stride = (grid // 12 + 1 + grid) * 32
    return State(
        torch.zeros((grid, 32), device=device, dtype=torch.int32),
        torch.zeros(phases * stride, device=device, dtype=torch.int32),
        torch.empty(grid, device=device, dtype=torch.int32),
        grid,
        stride,
        shape,
    )

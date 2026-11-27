"""GPU-local virtual-peer allocation and TLX setup; no physical peers."""

import msgspec
import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.plan import (
    RegionRequest,
    make_plan,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
)


def oneshot_plan(width, blocks_per_rank, world_size=4):
    return make_plan(
        world_size,
        (
            RegionRequest("payload", world_size * 4 * width),
            RegionRequest("flags", world_size * blocks_per_rank * 32),
        ),
    )


class Standins(msgspec.Struct):
    scratch: torch.Tensor
    peers: torch.Tensor
    ready: torch.Tensor
    reduced: torch.Tensor
    width: int
    ctas: int
    payload: tuple[int, int, int]
    flags: tuple[int, int, int]


def make_standins(device, *, width, ctas):
    plan = oneshot_plan(width, ctas, 4)
    scratch = torch.zeros(plan.total_words, device=device, dtype=torch.int32)
    regions = [plan.region(n) for n in ("payload", "flags")]
    return Standins(
        scratch,
        torch.full((4,), scratch.data_ptr(), device=device, dtype=torch.int64),
        torch.empty((1000, 4, ctas, 32), device=device, dtype=torch.int32),
        torch.empty((4, 8, width), device=device, dtype=torch.float32),
        width,
        ctas,
        *((r.base, r.rank_stride, r.parity_stride) for r in regions),
    )


def prepare(peers, reference, epochs, *, rank):
    assert reference.shape == (4, 8, peers.width) and reference.dtype == torch.bfloat16
    assert reference.is_contiguous() and 0 <= rank < 4
    chunks = triton.cdiv(4 * peers.width, 256)
    _prepare[4 * chunks,](
        reference.view(torch.int32),
        peers.scratch,
        epochs,
        peers.width,
        peers.ctas,
        rank,
        peers.payload,
        peers.flags,
        256,
        num_warps=4,
    )


@triton.jit
def _prepare(
    Reference,
    Scratch,
    Epochs,
    WIDTH: tl.constexpr,
    CTAS: tl.constexpr,
    RANK: tl.constexpr,
    PAYLOAD: tl.constexpr,
    FLAGS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    WORDS: tl.constexpr = 4 * WIDTH
    CHUNKS: tl.constexpr = tl.cdiv(WORDS, BLOCK)
    dst = tl.program_id(0) // CHUNKS
    block = tl.program_id(0) % CHUNKS
    i = tlx.require_layout(block * BLOCK + tl.arange(0, BLOCK), ROW1)
    tag = tl.load(Epochs) + 1
    parity = tag % 2
    for src in tl.static_range(4):
        if src != RANK:
            value = tl.load(Reference + src * WORDS + i, i < WORDS, 0)
            tl.store(
                Scratch
                + PAYLOAD[0]
                + dst * PAYLOAD[1]
                + parity * PAYLOAD[2]
                + src * WORDS
                + i,
                value,
                i < WORDS,
            )
            tl.store(
                Scratch
                + FLAGS[0]
                + dst * FLAGS[1]
                + parity * FLAGS[2]
                + (src * CTAS + i) * 32,
                tag,
                i < CTAS,
            )

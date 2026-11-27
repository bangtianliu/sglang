"""All-reduce of 8 bf16 rows across the four ranks through the chain's host-memory exchange.

One launch, one CTA per row: publish this rank's row, wait for the three peers (the host
exchange's deadline and hard stop apply), sum the four rows in rank order in fp32, round once to
bf16 and write the result over the input. It replaces the communication library's all-reduce for
the [8, 8192] and [8, 4096] tensors of a decode step outside the chain layers (layers 0 and 61,
the input embedding). The sum is rounded once, so it is at least as accurate as a pairwise bf16
sum but not bit-identical to it.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    host_shared,
    standin,
    tlx_memory,
    tlx_standin,
)


@triton.jit
def all_reduce_rows(X, Partial, Scratch, Peers, Reduced, Epochs,
                    GRID: tl.constexpr, RANK: tl.constexpr, PAYLOAD: tl.constexpr,
                    FLAGS: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    tag = tlx_standin.advance(Epochs, row)
    d = tl.arange(0, WIDTH)
    x = tl.load(X + row * WIDTH + d)
    tl.store(Partial + (RANK * 8 + row) * WIDTH + d, x)
    tlx_memory.drain_stores()
    value, ok = tlx_standin.reduce_row(
        Partial, Scratch, Peers, Reduced, tag, row, GRID, RANK, PAYLOAD, FLAGS,
        WIDTH=WIDTH, STORE_REDUCED=False,
    )
    tl.store(X + row * WIDTH + d, value.to(tl.bfloat16))


class HostAllReduce:
    """Per-rank state for one width; every rank constructs it at the same point."""

    GRID = 16

    def __init__(self, device, *, width: int, rank: int, barrier, name: str):
        assert tlx_standin.HOST.value, "needs MUSE_PEER_HOST=1"
        self.width, self.rank = width, rank
        self.peers = standin.make_standins(device, width=width, ctas=self.GRID)
        self.partial = torch.zeros((4, 8, width), device=device, dtype=torch.bfloat16)
        self.epochs = torch.zeros((8, 32), device=device, dtype=torch.int32)
        words = tlx_standin.host_words(width, self.GRID)
        self.host = host_shared.HostShared(f"{name}_ar{width}", words * 4, rank, barrier)
        if rank == 0:
            self.host.zero()
        self.peers.peers.fill_(self.host.dptr)
        barrier()

    def supported(self, x: torch.Tensor) -> bool:
        return (x.dim() == 2 and tuple(x.shape) == (8, self.width)
                and x.dtype == torch.bfloat16 and x.is_contiguous())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p = self.peers
        all_reduce_rows[(8,)](
            x, self.partial, p.scratch, p.peers, p.reduced, self.epochs,
            GRID=self.GRID, RANK=self.rank, PAYLOAD=p.payload, FLAGS=p.flags, WIDTH=self.width,
            num_warps=4, enable_fp_fusion=False,
        )
        return x

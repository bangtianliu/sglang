"""Reference stand-in peers for one real rank's GPU-local collective work.

prepare() is test setup outside the timed rank launch. It supplies the other
ranks' data and exact next-launch flags; the stage still executes its own puts,
polls, scoped reads and ascending-rank sum through the existing one-shot code.
"""

import msgspec
import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    vector_memory as vm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective_fast import (
    destination_pointers,
    fanout_flags,
    oneshot,
    oneshot_plan,
    sum_pair,
    wait_sources,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_tile_words,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.timeline import (
    stamp,
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


@g.jit
def _prepare(
    Reference,
    Scratch,
    Epochs,
    WIDTH: gl.constexpr,
    CTAS: gl.constexpr,
    RANK: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    BLOCK: gl.constexpr,
):
    WORDS: gl.constexpr = 4 * WIDTH
    CHUNKS: gl.constexpr = gl.cdiv(WORDS, BLOCK)
    dst = gl.program_id(0) // CHUNKS
    block = gl.program_id(0) % CHUNKS
    i = block * BLOCK + gl.arange(
        0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0])
    )
    tag = gl.load(Epochs) + 1
    parity = tag % 2
    for src in gl.static_range(4):
        if src != RANK:
            value = gl.load(Reference + src * WORDS + i, i < WORDS, 0)
            gl.store(
                Scratch
                + PAYLOAD[0]
                + dst * PAYLOAD[1]
                + parity * PAYLOAD[2]
                + src * WORDS
                + i,
                value,
                i < WORDS,
            )
            gl.store(
                Scratch
                + FLAGS[0]
                + dst * FLAGS[1]
                + parity * FLAGS[2]
                + (src * CTAS + i) * 32,
                tag,
                i < CTAS,
            )


def prepare(peers, reference, epochs, *, rank):
    assert reference.shape == (4, 8, peers.width) and reference.dtype == torch.bfloat16
    assert reference.is_contiguous() and 0 <= rank < 4
    chunks = triton.cdiv(4 * peers.width, 256)
    _prepare[(4 * chunks,)](
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


@g.jit
def allreduce_rank(
    Partial,
    Scratch,
    Peers,
    Ready,
    Reduced,
    tag,
    pid,
    WIDTH: gl.constexpr,
    GRID: gl.constexpr,
    RANK: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    Trace=None,
):
    oneshot(
        Partial.to(gl.pointer_type(gl.int32)),
        Scratch,
        Peers,
        None,
        Reduced,
        Ready,
        None,
        WIDTH,
        4,
        GRID,
        128,
        PAYLOAD,
        FLAGS,
        PROFILE=0,
        TAG=tag,
        SCOPED_INPUT=True,
        SCOPED_OUTPUT=True,
        READY_STRIDE=32,
        TASK_INDEX=RANK * GRID + pid,
        Trace=Trace,
    )
    scoped.drain_stores()
    return (
        scoped.load_flag(
            Ready + (((tag - 1) % 1000) * 4 * GRID + RANK * GRID + pid) * 32
        )
        == 1
    )


@g.jit
def reduce_row(
    Partial,
    Scratch,
    Peers,
    Reduced,
    tag,
    row,
    WIDTH: gl.constexpr,
    GRID: gl.constexpr,
    RANK: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    Trace=None,
):
    Q: gl.constexpr = WIDTH // 8
    q = gl.arange(0, Q, layout=gl.BlockedLayout([1], [32], [4], [0]))
    j = gl.arange(0, WIDTH // 2, layout=gl.BlockedLayout([4], [32], [4], [0]))
    at = row * (WIDTH // 2) + q * 4
    stamp(Trace, gl.program_id(0), 16)
    values = load_tile_words(
        Partial.to(gl.pointer_type(gl.int32))
        + RANK * 4 * WIDTH
        + row * (WIDTH // 2)
        + j,
        j >= 0,
        WIDTH // 2,
    )
    packets = vm.split4(values, Q)
    pointers = destination_pointers(Peers, at, tag % 2, RANK * 4 * WIDTH, PAYLOAD, 4)
    vm.store_destinations(pointers, packets, q >= 0, 4, min(4, Q // 128), COMMON=True)
    gl.barrier()
    fanout_flags(Peers, tag, tag % 2, RANK, row, FLAGS, 4, GRID)
    stamp(Trace, gl.program_id(0), 17)
    ready = wait_sources(Scratch, tag, tag % 2, RANK, row, FLAGS, 4, GRID, Trace)
    stamp(Trace, gl.program_id(0), 18)
    base = PAYLOAD[0] + RANK * PAYLOAD[1] + (tag % 2) * PAYLOAD[2]
    PACK: gl.constexpr = min(4, Q // 128)
    words = vm.load_sources(Scratch + base + at, q >= 0, 4 * WIDTH, 4, PACK)
    a = gl.full((Q,), 0, gl.float32, q.type.layout)
    b, c, d, e, f, g, h = a, a, a, a, a, a, a
    for src in gl.static_range(4):
        w0, w1, w2, w3 = (
            words[src * 4],
            words[src * 4 + 1],
            words[src * 4 + 2],
            words[src * 4 + 3],
        )
        a, b = sum_pair(a, b, w0)
        c, d = sum_pair(c, d, w1)
        e, f = sum_pair(e, f, w2)
        g, h = sum_pair(g, h, w3)
    value = vm.join8(a, b, c, d, e, f, g, h, Q)
    dindex = gl.arange(0, WIDTH, layout=value.type.layout)
    # Terminal debug/reference output; the boundary consumes value in registers.
    gl.store(Reduced + RANK * 8 * WIDTH + row * WIDTH + dindex, value)
    stamp(Trace, gl.program_id(0), 19)
    return value, ready

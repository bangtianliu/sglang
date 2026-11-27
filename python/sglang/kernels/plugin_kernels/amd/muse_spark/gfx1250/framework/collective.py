"""Parameterized virtual ranks: scoped reduce-scatter then all-gather on one GPU."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    mailbox,
    scoped,
    sync,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.plan import (
    RegionRequest,
    make_plan,
)


def allreduce_plan(width: int, blocks_per_rank: int, world_size: int = 4):
    """Allocate parity-buffered FP32 words and a private flag line per consumer."""
    if width <= 0 or blocks_per_rank <= 0 or world_size < 1 or (8 * width) % world_size:
        raise ValueError(
            "positive sizes and a world size dividing 8 * width are required"
        )
    return make_plan(
        world_size,
        (
            RegionRequest("partials", 8 * width),
            RegionRequest("partial_flags", world_size * blocks_per_rank * 32),
            RegionRequest("gathered", 8 * width),
            RegionRequest("gather_flags", world_size * blocks_per_rank * 32),
        ),
    )


@gluon.jit
def all_reduce(
    Inputs,
    Scratch,
    PeerBases,
    Epoch,
    Out,
    Ready,
    WIDTH: gl.constexpr,
    WORLD: gl.constexpr,
    BLOCKS_PER_RANK: gl.constexpr,
    BLOCK: gl.constexpr,
    PARTIALS: gl.constexpr,
    PARTIAL_FLAGS: gl.constexpr,
    GATHERED: gl.constexpr,
    GATHER_FLAGS: gl.constexpr,
    TIMES=None,
    PROFILE: gl.constexpr = False,
):
    """Sum BF16 [8,WIDTH] partials in ascending source-rank order and broadcast FP32 results."""
    pid = gl.program_id(0)
    rank = pid // BLOCKS_PER_RANK
    local = pid % BLOCKS_PER_RANK
    tag = gl.load(Epoch)
    parity = tag % 2
    gl.static_assert((8 * WIDTH) % WORLD == 0)
    SHARD: gl.constexpr = 8 * WIDTH // WORLD
    start = local * SHARD // BLOCKS_PER_RANK
    stop = (local + 1) * SHARD // BLOCKS_PER_RANK
    i = start + gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [32], [8], [0]))
    mask = i < stop
    if PROFILE:
        t0 = sync.clock()
    # Each source CTA pushes its partition of every destination's reduction shard.
    for dst in gl.static_range(WORLD):
        peer = gl.load(PeerBases + dst).to(gl.pointer_type(gl.int32))
        values = gl.load(
            Inputs + rank * (8 * WIDTH) + dst * SHARD + i, mask, other=0
        ).to(gl.float32)
        payload_offset = (
            PARTIALS[0] + dst * PARTIALS[1] + parity * PARTIALS[2] + rank * SHARD
        )
        flag_offset = (
            PARTIAL_FLAGS[0]
            + dst * PARTIAL_FLAGS[1]
            + parity * PARTIAL_FLAGS[2]
            + (rank * BLOCKS_PER_RANK + local) * 32
        )
        mailbox.put_words_peer(
            peer, payload_offset, i, values.to(gl.int32, bitcast=True), mask
        )
        mailbox.raise_flag_peer(peer, flag_offset, tag)
    if PROFILE:
        t1 = sync.clock()
    partial_base = PARTIALS[0] + rank * PARTIALS[1] + parity * PARTIALS[2]
    partial_flags = (
        PARTIAL_FLAGS[0] + rank * PARTIAL_FLAGS[1] + parity * PARTIAL_FLAGS[2]
    )
    acc = gl.full((BLOCK,), 0, gl.float32, gl.BlockedLayout([1], [32], [8], [0]))
    ready = True
    # Explicit left-to-right FP32 addition; no reassociation or BF16 rounding.
    for src in gl.static_range(WORLD):
        seen = mailbox.wait_flag(
            Scratch, partial_flags + (src * BLOCKS_PER_RANK + local) * 32, tag, 100_000
        )
        ready = ready & (seen == tag)
    if PROFILE:
        t2 = sync.clock()
    for src in gl.static_range(WORLD):
        words = scoped.load_words_masked(Scratch + partial_base + src * SHARD + i, mask)
        acc = acc + words.to(gl.float32, bitcast=True)
    if PROFILE:
        t3 = sync.clock()
    if ready:
        for dst in gl.static_range(WORLD):
            peer = gl.load(PeerBases + dst).to(gl.pointer_type(gl.int32))
            payload_offset = (
                GATHERED[0] + dst * GATHERED[1] + parity * GATHERED[2] + rank * SHARD
            )
            flag_offset = (
                GATHER_FLAGS[0]
                + dst * GATHER_FLAGS[1]
                + parity * GATHER_FLAGS[2]
                + (rank * BLOCKS_PER_RANK + local) * 32
            )
            mailbox.put_words_peer(
                peer, payload_offset, i, acc.to(gl.int32, bitcast=True), mask
            )
            mailbox.raise_flag_peer(peer, flag_offset, tag)
    if PROFILE:
        t4 = sync.clock()
    gathered = GATHERED[0] + rank * GATHERED[1] + parity * GATHERED[2]
    gather_flags = GATHER_FLAGS[0] + rank * GATHER_FLAGS[1] + parity * GATHER_FLAGS[2]
    for src in gl.static_range(WORLD):
        seen = mailbox.wait_flag(
            Scratch, gather_flags + (src * BLOCKS_PER_RANK + local) * 32, tag, 100_000
        )
        ready = ready & (seen == tag)
    if PROFILE:
        t5 = sync.clock()
    for src in gl.static_range(WORLD):
        words = scoped.load_words_masked(Scratch + gathered + src * SHARD + i, mask)
        gl.store(
            Out + rank * (8 * WIDTH) + src * SHARD + i,
            words.to(gl.float32, bitcast=True),
            mask,
        )
    if PROFILE:
        t6 = sync.clock()
        offset = (((tag - 1) % 1000) * (WORLD * BLOCKS_PER_RANK) + pid) * 6
        gl.store(TIMES + offset, t1 - t0)
        gl.store(TIMES + offset + 1, t2 - t1)
        gl.store(TIMES + offset + 2, t3 - t2)
        gl.store(TIMES + offset + 3, t4 - t3)
        gl.store(TIMES + offset + 4, t5 - t4)
        gl.store(TIMES + offset + 5, t6 - t5)
    gl.store(Ready + ((tag - 1) % 1000) * (WORLD * BLOCKS_PER_RANK) + pid, ready)

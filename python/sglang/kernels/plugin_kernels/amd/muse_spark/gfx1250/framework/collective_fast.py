"""Prefer one-round BF16 exchange; retain batched RS+AG as a tested alternative."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
    sync,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    vector_memory as vm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.plan import (
    RegionRequest,
    make_plan,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.timeline import (
    stamp,
)


def oneshot_plan(width, blocks_per_rank, world_size=4):
    return make_plan(
        world_size,
        (
            RegionRequest("payload", world_size * 4 * width),
            RegionRequest("flags", world_size * blocks_per_rank * 32),
        ),
    )


@gluon.jit
def destination_pointers(
    Peers, q, parity, source_offset, REGION: gl.constexpr, WORLD: gl.constexpr
):
    pointers = ()
    for dst in gl.static_range(WORLD):
        peer = gl.load(Peers + dst).to(gl.pointer_type(gl.int32))
        pointers += (
            peer + REGION[0] + dst * REGION[1] + parity * REGION[2] + source_offset + q,
        )
    return pointers


@gluon.jit
def fanout_flags(
    Peers,
    tag,
    parity,
    rank,
    local,
    FLAGS: gl.constexpr,
    WORLD: gl.constexpr,
    BPR: gl.constexpr,
):
    dst = gl.arange(
        0,
        gl.num_warps() * 32,
        layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]),
    )
    mask = dst < WORLD
    peer = gl.load(Peers + dst, mask, other=0).to(gl.pointer_type(gl.int32))
    offset = FLAGS[0] + dst * FLAGS[1] + parity * FLAGS[2] + (rank * BPR + local) * 32
    scoped.store_words(peer + offset, gl.full_like(dst, tag), mask)


@gluon.jit
def wait_sources(
    Scratch,
    tag,
    parity,
    rank,
    local,
    FLAGS: gl.constexpr,
    WORLD: gl.constexpr,
    BPR: gl.constexpr,
    Trace=None,
):
    src = gl.arange(
        0,
        gl.num_warps() * 32,
        layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]),
    )
    mask = src < WORLD
    base = FLAGS[0] + rank * FLAGS[1] + parity * FLAGS[2]
    ptr = Scratch + base + (src * BPR + local) * 32
    start = sync.clock()
    seen = scoped.load_words_masked(ptr, mask)
    ready = gl.sum(((seen == tag) & mask).to(gl.int32), 0) == WORLD
    first_mask = gl.sum(gl.where((seen == tag) & mask, 1 << src, 0), 0)
    spins = 1
    elapsed = start - start
    while (~ready) & (spins < 4096) & (elapsed < 100_000):
        seen = scoped.load_words_masked(ptr, mask)
        ready = gl.sum(((seen == tag) & mask).to(gl.int32), 0) == WORLD
        spins += 1
        if spins % 64 == 0:
            elapsed = sync.clock() - start
    gl.barrier()
    if Trace is not None:
        gl.store(Trace + gl.program_id(0) * 32 + 22, spins)
        gl.store(Trace + gl.program_id(0) * 32 + 23, first_mask)
    return ready


@gluon.jit
def batched_rs_ag(
    Inputs,
    Scratch,
    Peers,
    Epoch,
    Out,
    Ready,
    Times,
    WIDTH: gl.constexpr,
    WORLD: gl.constexpr,
    BPR: gl.constexpr,
    Q: gl.constexpr,
    PARTIALS: gl.constexpr,
    PARTIAL_FLAGS: gl.constexpr,
    GATHERED: gl.constexpr,
    GATHER_FLAGS: gl.constexpr,
    PROFILE: gl.constexpr = 2,
):
    pid = gl.program_id(0)
    rank, local = pid // BPR, pid % BPR
    tag = gl.load(Epoch)
    parity = tag % 2
    SHARD: gl.constexpr = 8 * WIDTH // WORLD
    gl.static_assert(SHARD % 4 == 0)
    PACK: gl.constexpr = max(1, Q // (32 * gl.num_warps()))
    first = local * (SHARD // 4) // BPR * 4
    last = (local + 1) * (SHARD // 4) // BPR * 4
    q = first + 4 * gl.arange(
        0, Q, layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    )
    mask = q < last
    j = first + gl.arange(
        0, 4 * Q, layout=gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
    )
    if PROFILE:
        t0 = sync.clock()
    layout: gl.constexpr = gl.BlockedLayout(
        [4, 1], [32, 1], [gl.num_warps(), 1], [0, 1]
    )
    rows = first + gl.arange(0, 4 * Q, layout=gl.SliceLayout(1, layout))
    columns = gl.arange(0, WORLD, layout=gl.SliceLayout(0, layout))
    values = (
        gl.load(
            Inputs + rank * 8 * WIDTH + rows[:, None] + columns[None, :] * SHARD,
            rows[:, None] < last,
            other=0,
        )
        .to(gl.float32)
        .to(gl.int32, bitcast=True)
    )
    sources = vm.split_world(values, 4 * Q, WORLD)
    packets = ()
    for dst in gl.static_range(WORLD):
        values = gl.convert_layout(
            sources[dst], gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
        )
        packets += vm.split4(values, Q)
    pointers = destination_pointers(Peers, q, parity, rank * SHARD, PARTIALS, WORLD)
    vm.store_destinations(pointers, packets, mask, WORLD, PACK)
    gl.barrier()
    fanout_flags(Peers, tag, parity, rank, local, PARTIAL_FLAGS, WORLD, BPR)
    if PROFILE == 1:
        t1 = sync.clock()
    ready = wait_sources(Scratch, tag, parity, rank, local, PARTIAL_FLAGS, WORLD, BPR)
    if PROFILE == 1:
        t2 = sync.clock()
    base = PARTIALS[0] + rank * PARTIALS[1] + parity * PARTIALS[2]
    words = vm.load_sources(Scratch + base + q, mask, SHARD, WORLD, PACK)
    a = gl.full((Q,), 0, gl.float32, gl.BlockedLayout([1], [32], [gl.num_warps()], [0]))
    b, c, d = a, a, a
    for src in gl.static_range(WORLD):
        a = a + words[src * 4].to(gl.float32, bitcast=True)
        b = b + words[src * 4 + 1].to(gl.float32, bitcast=True)
        c = c + words[src * 4 + 2].to(gl.float32, bitcast=True)
        d = d + words[src * 4 + 3].to(gl.float32, bitcast=True)
    if PROFILE == 1:
        t3 = sync.clock()
    if ready:
        pointers = destination_pointers(Peers, q, parity, rank * SHARD, GATHERED, WORLD)
        vm.store_destinations(
            pointers,
            (
                a.to(gl.int32, bitcast=True),
                b.to(gl.int32, bitcast=True),
                c.to(gl.int32, bitcast=True),
                d.to(gl.int32, bitcast=True),
            ),
            mask,
            WORLD,
            PACK,
            COMMON=True,
        )
        gl.barrier()
        fanout_flags(Peers, tag, parity, rank, local, GATHER_FLAGS, WORLD, BPR)
    if PROFILE == 1:
        t4 = sync.clock()
    ready = ready & wait_sources(
        Scratch, tag, parity, rank, local, GATHER_FLAGS, WORLD, BPR
    )
    if PROFILE == 1:
        t5 = sync.clock()
    base = GATHERED[0] + rank * GATHERED[1] + parity * GATHERED[2]
    words = vm.load_sources(Scratch + base + q, mask, SHARD, WORLD, PACK)
    for src in gl.static_range(WORLD):
        values = vm.join4(
            words[src * 4],
            words[src * 4 + 1],
            words[src * 4 + 2],
            words[src * 4 + 3],
            Q,
        ).to(gl.float32, bitcast=True)
        gl.store(Out + rank * 8 * WIDTH + src * SHARD + j, values, j < last)
    if PROFILE:
        t6 = sync.clock()
        at = (((tag - 1) % 1000) * (WORLD * BPR) + pid) * 7
        gl.store(Times + at + 6, t6 - t0)
        if PROFILE == 1:
            gl.store(Times + at, t1 - t0)
            gl.store(Times + at + 1, t2 - t1)
            gl.store(Times + at + 2, t3 - t2)
            gl.store(Times + at + 3, t4 - t3)
            gl.store(Times + at + 4, t5 - t4)
            gl.store(Times + at + 5, t6 - t5)
    gl.store(Ready + ((tag - 1) % 1000) * (WORLD * BPR) + pid, ready)


@gluon.jit
def sum_pair(lo, hi, word):
    a = (word & 65535).to(gl.uint16).to(gl.bfloat16, bitcast=True).to(gl.float32)
    b = (word >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True).to(gl.float32)
    return lo + a, hi + b


@gluon.jit
def oneshot(
    InputsPacked,
    Scratch,
    Peers,
    Epoch,
    Out,
    Ready,
    Times,
    WIDTH: gl.constexpr,
    WORLD: gl.constexpr,
    BPR: gl.constexpr,
    Q: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    PROFILE: gl.constexpr = 2,
    TAG=None,
    SCOPED_INPUT: gl.constexpr = False,
    SCOPED_OUTPUT: gl.constexpr = False,
    READY_STRIDE: gl.constexpr = 1,
    TASK_INDEX=None,
    Trace=None,
):
    pid = gl.program_id(0) if TASK_INDEX is None else TASK_INDEX
    rank, local = pid // BPR, pid % BPR
    tag = gl.load(Epoch) if TAG is None else TAG
    parity = tag % 2
    WORDS: gl.constexpr = 4 * WIDTH
    PACK: gl.constexpr = max(1, Q // (32 * gl.num_warps()))
    first = local * (WORDS // 4) // BPR * 4
    last = (local + 1) * (WORDS // 4) // BPR * 4
    q = first + 4 * gl.arange(
        0, Q, layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    )
    mask = q < last
    j = first + gl.arange(
        0, 4 * Q, layout=gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
    )
    if PROFILE:
        t0 = sync.clock()
    stamp(Trace, gl.program_id(0), 16)
    if SCOPED_INPUT:
        values = scoped.load_words_masked(InputsPacked + rank * WORDS + j, j < last)
    else:
        values = gl.load(InputsPacked + rank * WORDS + j, j < last, other=0)
    packets = vm.split4(values, Q)
    pointers = destination_pointers(Peers, q, parity, rank * WORDS, PAYLOAD, WORLD)
    vm.store_destinations(pointers, packets, mask, WORLD, PACK, COMMON=True)
    gl.barrier()
    fanout_flags(Peers, tag, parity, rank, local, FLAGS, WORLD, BPR)
    if PROFILE == 1:
        t1 = sync.clock()
    stamp(Trace, gl.program_id(0), 17)
    ready = wait_sources(Scratch, tag, parity, rank, local, FLAGS, WORLD, BPR, Trace)
    stamp(Trace, gl.program_id(0), 18)
    if PROFILE == 1:
        t2 = sync.clock()
    offset = PAYLOAD[0] + rank * PAYLOAD[1] + parity * PAYLOAD[2]
    words = vm.load_sources(Scratch + offset + q, mask, WORDS, WORLD, PACK)
    a = gl.full((Q,), 0, gl.float32, gl.BlockedLayout([1], [32], [gl.num_warps()], [0]))
    b, c, d, e, f, g, h = a, a, a, a, a, a, a
    for src in gl.static_range(WORLD):
        a, b = sum_pair(a, b, words[src * 4])
        c, d = sum_pair(c, d, words[src * 4 + 1])
        e, f = sum_pair(e, f, words[src * 4 + 2])
        g, h = sum_pair(g, h, words[src * 4 + 3])
    values = vm.join8(a, b, c, d, e, f, g, h, Q)
    indices = first * 2 + gl.arange(
        0, 8 * Q, layout=gl.BlockedLayout([8], [32], [gl.num_warps()], [0])
    )
    if SCOPED_OUTPUT:
        scoped.store_words(
            Out.to(gl.pointer_type(gl.int32)) + rank * 8 * WIDTH + indices,
            values.to(gl.int32, bitcast=True),
            indices < last * 2,
        )
    else:
        gl.store(Out + rank * 8 * WIDTH + indices, values, indices < last * 2)
    stamp(Trace, gl.program_id(0), 19)
    if PROFILE:
        t3 = sync.clock()
        at = (((tag - 1) % 1000) * (WORLD * BPR) + pid) * 7
        gl.store(Times + at + 6, t3 - t0)
        if PROFILE == 1:
            gl.store(Times + at, t1 - t0)
            gl.store(Times + at + 1, t2 - t1)
            gl.store(Times + at + 2, t3 - t2)
    if SCOPED_OUTPUT:
        scoped.store_words(
            Ready + (((tag - 1) % 1000) * (WORLD * BPR) + pid) * READY_STRIDE,
            ready.to(gl.int32),
            pid >= 0,
        )
    else:
        gl.store(
            Ready + (((tag - 1) % 1000) * (WORLD * BPR) + pid) * READY_STRIDE, ready
        )

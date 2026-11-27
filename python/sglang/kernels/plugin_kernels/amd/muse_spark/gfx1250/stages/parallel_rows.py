"""Tiled row epilogues for eight-row gfx1250 layers, four waves, fixed FP32 tree.

Boundary width8192 and pool width4096 only, on the pinned Triton3.9 stack.
Private per-consumer flags and bounded shader-cycle waits; callers admit a
resident cooperative grid before launch. No physical peer mapping is created.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    vector_memory as vm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.timeline import (
    stamp,
)


@g.jit
def columns(tile, WIDTH: gl.constexpr):
    # LLVM vectorizes pairs of registers, reducing bits1,2,... before bit0.
    # Reduce bits1..3 here and leave bit0/higher bits for the factor owner.
    l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    hi = gl.arange(0, 8, layout=gl.SliceLayout(1, l))
    lane = gl.arange(0, 8, layout=gl.SliceLayout(0, l))
    low = tile // 16
    start = (tile % 16) * 8
    return (hi[:, None] * 2 + (low % 2) + (low // 2) * 16) * 128 + start + lane[None, :]


@g.jit
def partial_stat(Value):
    square = gl.permute(Value * Value, (1, 0))
    even, odd = gl.split(square.reshape((8, 4, 2)))
    a0, a2, a4, a6 = vm.split4(even.reshape((32,)), 8)
    a1, a3, a5, a7 = vm.split4(odd.reshape((32,)), 8)
    return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))


@g.jit
def factor(Stats, row, WIDTH: gl.constexpr, EPS: gl.constexpr):
    l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    low = gl.arange(0, WIDTH // 1024, layout=gl.SliceLayout(1, l))
    lane = gl.arange(0, 128, layout=gl.SliceLayout(0, l))
    v = gl.load(
        Stats + row * (WIDTH // 8) + low[:, None] * 128 + lane[None, :],
        cache_modifier=".cg",
    )
    total = gl.sum(gl.sum(v, 0), 0)
    return total, gl.rsqrt(total / WIDTH + EPS)


import torch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    layer_flow as flow,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
    sync,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective_fast import (
    destination_pointers,
    sum_pair,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_tile_words,
    store_bf16_tile,
    store_tile_words,
)


@g.constexpr_function
def row_plan(width):
    # B, stats0/1, factors0/1, counters0/1, ready, flags0/1, done, join, total.
    tasks = 8 * (width // 64)
    ready = 10 * width + 1024
    return (
        0,
        8 * width,
        9 * width,
        10 * width,
        10 * width + 256,
        10 * width + 512,
        10 * width + 768,
        ready,
        ready + tasks * 32,
        ready + tasks * 64,
        ready + tasks * 96,
        ready + tasks * 96 + 256,
        ready + tasks * 96 + 288,
    )


def make_state(device, width):
    return torch.zeros(row_plan(width)[12], device=device, dtype=torch.int32)


@g.jit
def wait_line(
    Flag,
    tag,
    pid,
    task,
    Diag,
    STAGE: gl.constexpr,
    Record=None,
    SLOT: gl.constexpr = 20,
):
    seen, polls = scoped.wait_flag(
        Flag, tag, 100_000_000, MAX_SPINS=1_048_576, CLOCK_EVERY=64, SHADER_CLOCK=True
    )
    if Record is not None:
        gl.store(Record + task * 32 + SLOT, polls)
    if seen != tag:
        flow.note(Diag, pid, STAGE, STAGE, task, tag, seen, polls)
    return seen == tag


@g.jit
def publish_stat(
    State,
    Value,
    row,
    tile,
    tag,
    WIDTH: gl.constexpr,
    PHASE: gl.constexpr,
    EPS: gl.constexpr,
):
    p: gl.constexpr = row_plan(WIDTH)
    s = partial_stat(Value)
    j = gl.arange(0, 8, layout=s.type.layout)
    at = p[1 + PHASE] + row * (WIDTH // 8) + (tile // 16) * 128 + (tile % 16) * 8 + j
    scoped.store_words(State + at, s.to(gl.int32, bitcast=True), j >= 0)
    if flow.arrive(State + p[5 + PHASE] + row * 32, tag, WIDTH // 64):
        _, inv = factor(
            (State + p[1 + PHASE]).to(gl.pointer_type(gl.float32)), row, WIDTH, EPS
        )
        scoped.store_words(
            State + p[3 + PHASE] + row * 32,
            inv.to(gl.int32, bitcast=True),
            gl.full((), True, gl.int1, j.type.layout),
        )
        scoped.publish_flags(
            State + p[8 + PHASE] + row * (WIDTH // 64) * 32, tag, WIDTH // 64
        )


@g.jit
def publish_ready(State, tag, WIDTH: gl.constexpr):
    p: gl.constexpr = row_plan(WIDTH)
    scoped.publish_flags(State + p[7], tag, 8 * (WIDTH // 64))


@g.jit
def put_tile(
    Value, tile, Peers, tag, RANK: gl.constexpr, PAY: gl.constexpr, WIDTH: gl.constexpr
):
    l: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [4, 1], [1, 0])
    v = gl.convert_layout(Value.to(gl.bfloat16), l)
    lo, hi = gl.split(v.reshape((8, 32, 2)))
    bits = (
        lo.to(gl.uint16, bitcast=True).to(gl.uint32)
        | (hi.to(gl.uint16, bitcast=True).to(gl.uint32) << 16)
    ).to(gl.int32)
    packets = vm.split4(
        gl.convert_layout(bits.reshape((256,)), gl.BlockedLayout([4], [32], [4], [0])),
        64,
    )
    q = gl.arange(0, 64, layout=gl.BlockedLayout([1], [32], [4], [0]))
    at = (q // 8) * (WIDTH // 2) + tile * 32 + (q % 8) * 4
    ptr = destination_pointers(Peers, at, tag % 2, RANK * 4 * WIDTH, PAY, 4)
    vm.store_destinations(ptr, packets, q >= 0, 4, 1, COMMON=True)


@g.jit
def publish_sources(
    Peers, tag, RANK: gl.constexpr, FL: gl.constexpr, CTAS: gl.constexpr
):
    for dst in gl.static_range(4):
        peer = gl.load(Peers + dst).to(gl.pointer_type(gl.int32))
        scoped.publish_flags(
            peer + FL[0] + dst * FL[1] + (tag % 2) * FL[2] + RANK * CTAS * 32, tag, CTAS
        )


@g.jit
def sum_fragment(
    Scratch,
    row,
    tile,
    tag,
    RANK: gl.constexpr,
    PAY: gl.constexpr,
    FL: gl.constexpr,
    WIDTH: gl.constexpr,
):
    # One b128 packet per high-register group (eight adjacent BF16 columns).
    q = gl.arange(0, 8, layout=gl.BlockedLayout([1], [32], [4], [0]))
    low = tile // 16
    first = (tile % 16) * 8
    col = (q * 2 + low % 2 + (low // 2) * 16) * 128 + first
    base = PAY[0] + RANK * PAY[1] + (tag % 2) * PAY[2]
    words = vm.load_sources(
        Scratch + base + row * (WIDTH // 2) + col // 2, q >= 0, 4 * WIDTH, 4, 1
    )
    a = gl.full((8,), 0, gl.float32, q.type.layout)
    b, c, d, e, f, h, i = a, a, a, a, a, a, a
    for src in gl.static_range(4):
        a, b = sum_pair(a, b, words[src * 4])
        c, d = sum_pair(c, d, words[src * 4 + 1])
        e, f = sum_pair(e, f, words[src * 4 + 2])
        h, i = sum_pair(h, i, words[src * 4 + 3])
    v = vm.join8(a, b, c, d, e, f, h, i, 8).reshape((8, 8))
    l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    return gl.convert_layout(v, l)


@g.jit
def wait_sources(
    Scratch,
    tag,
    task,
    pid,
    Diag,
    RANK: gl.constexpr,
    FL: gl.constexpr,
    CTAS: gl.constexpr,
    Record=None,
):
    ids = gl.arange(0, 4, layout=gl.BlockedLayout([1], [32], [4], [0]))
    ptr = Scratch + FL[0] + RANK * FL[1] + (tag % 2) * FL[2] + (ids * CTAS + task) * 32
    start = sync.shader_cycles()
    seen = scoped.load_words_masked(ptr, ids >= 0)
    ready = gl.sum((seen == tag).to(gl.int32), 0) == 4
    spins = 1
    elapsed = start - start
    while (~ready) & (spins < 1_048_576) & (elapsed < 100_000_000):
        seen = scoped.load_words_masked(ptr, ids >= 0)
        ready = gl.sum((seen == tag).to(gl.int32), 0) == 4
        spins += 1
        if spins % 64 == 0:
            elapsed = sync.shader_cycles() - start
    gl.barrier()
    if Record is not None:
        gl.store(Record + task * 32 + 20, spins)
    if not ready:
        flow.note(Diag, pid, 3, 3, task, tag, 0, spins)
    return ready


@g.jit
def boundary(
    pid,
    tag,
    State,
    Branch,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    DoneCounter,
    DoneFlags,
    Diag,
    Metrics,
    GRID: gl.constexpr,
    STAGE: gl.constexpr,
    AR: gl.constexpr = False,
    Scratch=None,
    RANK: gl.constexpr = 0,
    PAY: gl.constexpr = (),
    FL: gl.constexpr = (),
    Trace=None,
    Record=None,
):
    WIDTH: gl.constexpr = 8192
    TILES: gl.constexpr = 128
    TASKS: gl.constexpr = 1024
    p: gl.constexpr = row_plan(WIDTH)
    ok = gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    stamp(Trace, pid, 0)
    # Produce every assigned partial before waiting for any normalization factor.
    for task in range(pid, TASKS, GRID):
        row = task // TILES
        tile = task % TILES
        c = columns(tile, WIDTH)
        stamp(Record, task, 0)
        if AR:
            ready = wait_sources(Scratch, tag, task, pid, Diag, RANK, FL, TASKS, Record)
        else:
            ready = wait_line(
                State + p[7] + task * 32, tag, pid, task, Diag, STAGE, Record, 20
            )
        stamp(Record, task, 1)
        ok = ok & ready
        if ready:
            if AR:
                v = sum_fragment(Scratch, row, tile, tag, RANK, PAY, FL, WIDTH)
            else:
                v = load_tile_words(Branch + row * WIDTH + c, c >= 0, 64).to(
                    gl.float32, bitcast=True
                )
            v = v.to(gl.bfloat16).to(gl.float32)
            if Post is not None:
                v = (
                    (v * gl.load(Post + c).to(gl.float32))
                    .to(gl.bfloat16)
                    .to(gl.float32)
                )
            store_tile_words(
                State + row * WIDTH + c, v.to(gl.int32, bitcast=True), c >= 0, 64
            )
            stamp(Record, task, 2)
            publish_stat(State, v, row, tile, tag, WIDTH, 0, 1e-8)
            stamp(Record, task, 3)
        gl.store(
            Metrics + pid * 32 + 15 + STAGE,
            gl.load(Metrics + pid * 32 + 15 + STAGE) + 1,
        )
    stamp(Trace, pid, 1)
    for task in range(pid, TASKS, GRID):
        row = task // TILES
        tile = task % TILES
        c = columns(tile, WIDTH)
        stamp(Record, task, 4)
        ready = wait_line(
            State + p[8] + task * 32, tag, pid, task, Diag, STAGE, Record, 21
        )
        stamp(Record, task, 5)
        ok = ok & ready
        if ready:
            v = load_tile_words(State + row * WIDTH + c, c >= 0, 64).to(
                gl.float32, bitcast=True
            )
            inv = scoped.load_flag(State + p[3] + row * 32).to(gl.float32, bitcast=True)
            norm = (v * inv).to(gl.bfloat16).to(gl.float32)
            stream = gl.load(Stream + row * WIDTH + c, cache_modifier=".cg")
            updated = gl.load(Alpha + c) * stream + gl.load(Beta + c) * norm
            store_tile_words(
                Updated.to(gl.pointer_type(gl.int32)) + row * WIDTH + c,
                updated.to(gl.int32, bitcast=True),
                c >= 0,
                64,
            )
            stamp(Record, task, 6)
            publish_stat(State, updated, row, tile, tag, WIDTH, 1, 1e-5)
            stamp(Record, task, 7)
    stamp(Trace, pid, 2)
    for task in range(pid, TASKS, GRID):
        row = task // TILES
        tile = task % TILES
        c = columns(tile, WIDTH)
        stamp(Record, task, 8)
        ready = wait_line(
            State + p[9] + task * 32, tag, pid, task, Diag, STAGE, Record, 22
        )
        stamp(Record, task, 9)
        ok = ok & ready
        if ready:
            v = load_tile_words(
                Updated.to(gl.pointer_type(gl.int32)) + row * WIDTH + c, c >= 0, 64
            ).to(gl.float32, bitcast=True)
            inv = scoped.load_flag(State + p[4] + row * 32).to(gl.float32, bitcast=True)
            v = v * inv
            if Pre is not None:
                v = v * gl.load(Pre + c).to(gl.float32)
            store_bf16_tile(Out + row * WIDTH + c, v, c >= 0, 8, 8)
            if DoneCounter is not None:
                if flow.arrive(State + p[10] + row * 32, tag, TILES):
                    if flow.arrive(DoneCounter, tag, 8):
                        scoped.publish_flags(DoneFlags, tag, 576)
        stamp(Record, task, 10)
    stamp(Trace, pid, 3)
    return ok


@g.jit
def pool(
    pid,
    tag,
    State,
    Projection,
    Norm,
    Out,
    Join,
    Flags,
    Diag,
    Metrics,
    GRID: gl.constexpr,
    Trace=None,
    Record=None,
):
    WIDTH: gl.constexpr = 4096
    TILES: gl.constexpr = 64
    TASKS: gl.constexpr = 512
    p: gl.constexpr = row_plan(WIDTH)
    ok = gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    stamp(Trace, pid, 0)
    for task in range(pid, TASKS, GRID):
        row = task // TILES
        tile = task % TILES
        c = columns(tile, WIDTH)
        stamp(Record, task, 0)
        ready = wait_line(State + p[7] + task * 32, tag, pid, task, Diag, 4, Record, 20)
        stamp(Record, task, 1)
        ok = ok & ready
        if ready:
            v = (
                load_tile_words(Projection + row * 4608 + c, c >= 0, 64)
                .to(gl.float32, bitcast=True)
                .to(gl.bfloat16)
                .to(gl.float32)
            )
            store_tile_words(
                State + row * WIDTH + c, v.to(gl.int32, bitcast=True), c >= 0, 64
            )
            stamp(Record, task, 2)
            publish_stat(State, v, row, tile, tag, WIDTH, 0, 1e-5)
            stamp(Record, task, 3)
        gl.store(Metrics + pid * 32 + 19, gl.load(Metrics + pid * 32 + 19) + 1)
    stamp(Trace, pid, 1)
    first = (pid + 512) % GRID
    for task in range(first, TASKS, GRID):
        row = task // TILES
        tile = task % TILES
        c = columns(tile, WIDTH)
        stamp(Record, task, 4)
        ready = wait_line(State + p[8] + task * 32, tag, pid, task, Diag, 4, Record, 21)
        stamp(Record, task, 5)
        ok = ok & ready
        if ready:
            v = load_tile_words(State + row * WIDTH + c, c >= 0, 64).to(
                gl.float32, bitcast=True
            )
            inv = scoped.load_flag(State + p[3] + row * 32).to(gl.float32, bitcast=True)
            v = v * inv
            v = v * gl.load(Norm + c).to(gl.float32)
            store_bf16_tile(Out + row * WIDTH + c, v, c >= 0, 8, 8)
            if flow.arrive(State + p[6] + row * 32, tag, TILES):
                if flow.arrive(State + p[10], tag, 8):
                    if flow.arrive(Join, tag, 2):
                        scoped.publish_flags(Flags, tag, GRID)
        gl.store(Metrics + pid * 32 + 19, gl.load(Metrics + pid * 32 + 19) + 1)
        stamp(Record, task, 6)
    stamp(Trace, pid, 2)
    return ok

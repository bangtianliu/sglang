"""Bounded 4.4c producer statistics and exact-tree row epilogue experiments.

A squared-value packet is deliberately unreduced: the accepted sum tree joins
columns from different contiguous GEMM tiles before adjacent columns. Reducing
within a contiguous tile would change that tree. All VMEM uses audited helpers.
"""

import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    layer_flow as flow,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    vector_memory as vm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective_fast import (
    sum_pair,
    wait_sources,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_tile_words,
    store_bf16_vector,
    store_tile_words,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.timeline import (
    stamp,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.parallel_rows import (
    put_tile,
    wait_line,
)


@g.constexpr_function
def plan(width):
    # activation B, squared packets0/1, factors0/1, counters0/1, private flags0/1,
    # one completion flag per row (one forwarding consumer), final row counters.
    return (
        0,
        8 * width,
        16 * width,
        24 * width,
        24 * width + 256,
        24 * width + 512,
        24 * width + 768,
        24 * width + 1024,
        24 * width + 1024 + 256 * 32,
        24 * width + 1024 + 512 * 32,
        24 * width + 1024 + 520 * 32,
        24 * width + 1024 + 528 * 32,
    )


def make_state(device, width):
    return torch.zeros(plan(width)[11] + 32, device=device, dtype=torch.int32)


@g.jit
def row_factor(State, row, WIDTH: gl.constexpr, PHASE: gl.constexpr, EPS: gl.constexpr):
    p: gl.constexpr = plan(WIDTH)
    i = gl.arange(0, WIDTH, layout=gl.BlockedLayout([1], [32], [4], [0]))
    square = load_tile_words(State + p[1 + PHASE] + row * WIDTH + i, i >= 0, WIDTH).to(
        gl.float32, bitcast=True
    )
    return gl.rsqrt(gl.sum(square, 0) / WIDTH + EPS)


@g.jit
def collective(
    Value,
    tile,
    Peers,
    Scratch,
    tag,
    RANK: gl.constexpr,
    PAY: gl.constexpr,
    FL: gl.constexpr,
    Record=None,
):
    put_tile(Value, tile, Peers, tag, RANK, PAY, 8192)
    scoped.drain_stores()
    for dst in gl.static_range(4):
        ptr = gl.load(Peers + dst).to(gl.pointer_type(gl.int32))
        scoped.publish(
            ptr + FL[0] + dst * FL[1] + (tag % 2) * FL[2] + (RANK * 128 + tile) * 32,
            tag,
        )
    # One private consumer per projection tile; stand-in peers are already ready.
    ready = wait_sources(Scratch, tag, tag % 2, RANK, tile, FL, 4, 128, Record)
    q = gl.arange(0, 64, layout=gl.BlockedLayout([1], [32], [4], [0]))
    at = (q // 8) * 4096 + tile * 32 + (q % 8) * 4
    base = PAY[0] + RANK * PAY[1] + (tag % 2) * PAY[2]
    words = vm.load_sources(Scratch + base + at, q >= 0, 4 * 8192, 4, 1)
    a = gl.full((64,), 0, gl.float32, q.type.layout)
    b, c, d, e, f, h, j = a, a, a, a, a, a, a
    for src in gl.static_range(4):
        a, b = sum_pair(a, b, words[src * 4])
        c, d = sum_pair(c, d, words[src * 4 + 1])
        e, f = sum_pair(e, f, words[src * 4 + 2])
        h, j = sum_pair(h, j, words[src * 4 + 3])
    value = vm.join8(a, b, c, d, e, f, h, j, 64).reshape((8, 64))
    return value, ready


@g.jit
def finish_row(
    State,
    row,
    tag,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    WIDTH: gl.constexpr,
    POOL: gl.constexpr,
    Trace=None,
):
    stamp(Trace, 0, 0)
    p: gl.constexpr = plan(WIDTH)
    i = gl.arange(0, WIDTH, layout=gl.BlockedLayout([1], [32], [4], [0]))
    v = load_tile_words(State + row * WIDTH + i, i >= 0, WIDTH).to(
        gl.float32, bitcast=True
    )
    stamp(Trace, 0, 3)
    inv = row_factor(State, row, WIDTH, 0, 1e-5 if POOL else 1e-8)
    stamp(Trace, 0, 5)
    if POOL:
        out = v * inv * gl.load(Pre + i).to(gl.float32)
    else:
        norm = (v * inv).to(gl.bfloat16).to(gl.float32)
        stream = gl.load(Stream + row * WIDTH + i, cache_modifier=".cg")
        alpha = gl.load(Alpha + i)
        beta = gl.load(Beta + i)
        stamp(Trace, 0, 11)
        updated = alpha * stream + beta * norm
        stamp(Trace, 0, 12)
        store_tile_words(
            Updated.to(gl.pointer_type(gl.int32)) + row * WIDTH + i,
            updated.to(gl.int32, bitcast=True),
            i >= 0,
            WIDTH,
        )
        stamp(Trace, 0, 6)
        total = gl.sum(updated * updated, 0)
        stamp(Trace, 0, 13)
        out = updated * gl.rsqrt(total / WIDTH + 1e-5)
        gain = gl.load(Pre + i).to(gl.float32)
        stamp(Trace, 0, 14)
        out = out * gain
        stamp(Trace, 0, 7)
    store_bf16_vector(Out + row * WIDTH, out, WIDTH)
    stamp(Trace, 0, 8)
    scoped.publish(State + p[9] + row * 32, tag)
    stamp(Trace, 0, 15)


@g.jit
def produce(
    Value,
    tile,
    State,
    tag,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    WIDTH: gl.constexpr,
    MODE: gl.constexpr,
    POOL: gl.constexpr,
    Trace=None,
):
    p: gl.constexpr = plan(WIDTH)
    l: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    v = gl.convert_layout(Value, l).to(gl.bfloat16).to(gl.float32)
    r = gl.arange(0, 8, layout=gl.SliceLayout(1, l))
    c = tile * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, l))
    if Post is not None:
        v = (
            (v * gl.load(Post + c[None, :]).to(gl.float32))
            .to(gl.bfloat16)
            .to(gl.float32)
        )
    store_tile_words(
        State + r[:, None] * WIDTH + c[None, :],
        v.to(gl.int32, bitcast=True),
        r[:, None] >= 0,
        512,
    )
    store_tile_words(
        State + p[1] + r[:, None] * WIDTH + c[None, :],
        (v * v).to(gl.int32, bitcast=True),
        r[:, None] >= 0,
        512,
    )
    for rr in range(8):
        row = (rr + tile) % 8
        if flow.arrive(State + p[5] + row * 32, tag, WIDTH // 64):
            if MODE == 2:
                stamp(Trace, gl.program_id(0), 16 + row)
                finish_row(
                    State,
                    row,
                    tag,
                    Stream,
                    Post,
                    Pre,
                    Alpha,
                    Beta,
                    Out,
                    Updated,
                    WIDTH,
                    POOL,
                )
                stamp(Trace, gl.program_id(0), 24 + row)
            else:
                inv = row_factor(State, row, WIDTH, 0, 1e-5 if POOL else 1e-8)
                scoped.store_words(
                    State + p[3] + row * 32,
                    inv.to(gl.int32, bitcast=True),
                    gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0])),
                )
                chunks: gl.constexpr = WIDTH // (256 if POOL else 512)
                scoped.publish_flags(State + p[7] + row * chunks * 32, tag, chunks)


@g.jit
def apply(
    pid,
    tag,
    State,
    Stream,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    Diag,
    Metrics,
    WIDTH: gl.constexpr,
    MODE: gl.constexpr,
    POOL: gl.constexpr,
    STAGE: gl.constexpr,
    Counter,
    Flags,
    NEXT: gl.constexpr,
    Trace=None,
    Record=None,
    Join=None,
):
    p: gl.constexpr = plan(WIDTH)
    CHUNK: gl.constexpr = 256 if POOL else 512
    CHUNKS: gl.constexpr = WIDTH // CHUNK
    TASKS: gl.constexpr = 8 * CHUNKS
    ok = gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    stamp(Trace, pid, 0)
    if MODE == 1:
        task = pid - 512
        if task >= 0 and task < TASKS:
            ready = wait_line(
                State + p[7] + task * 32, tag, pid, task, Diag, STAGE, Record, 20
            )
            ok = ok & ready
            if ready:
                row = task // CHUNKS
                i = (task % CHUNKS) * CHUNK + gl.arange(
                    0, CHUNK, layout=gl.BlockedLayout([1], [32], [4], [0])
                )
                v = load_tile_words(State + row * WIDTH + i, i >= 0, CHUNK).to(
                    gl.float32, bitcast=True
                )
                inv = scoped.load_flag(State + p[3] + row * 32).to(
                    gl.float32, bitcast=True
                )
                if POOL:
                    store_bf16_vector(
                        Out + row * WIDTH + (task % CHUNKS) * CHUNK,
                        v * inv * gl.load(Pre + i).to(gl.float32),
                        CHUNK,
                    )
                    if flow.arrive(State + p[10] + row * 32, tag, CHUNKS):
                        scoped.publish(State + p[9] + row * 32, tag)
                else:
                    norm = (v * inv).to(gl.bfloat16).to(gl.float32)
                    updated = (
                        gl.load(Alpha + i)
                        * gl.load(Stream + row * WIDTH + i, cache_modifier=".cg")
                        + gl.load(Beta + i) * norm
                    )
                    store_tile_words(
                        Updated.to(gl.pointer_type(gl.int32)) + row * WIDTH + i,
                        updated.to(gl.int32, bitcast=True),
                        i >= 0,
                        CHUNK,
                    )
                    store_tile_words(
                        State + p[2] + row * WIDTH + i,
                        (updated * updated).to(gl.int32, bitcast=True),
                        i >= 0,
                        CHUNK,
                    )
                    if flow.arrive(State + p[6] + row * 32, tag, CHUNKS):
                        inv = row_factor(State, row, WIDTH, 1, 1e-5)
                        scoped.store_words(
                            State + p[4] + row * 32,
                            inv.to(gl.int32, bitcast=True),
                            gl.full(
                                (), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0])
                            ),
                        )
                        scoped.publish_flags(
                            State + p[8] + row * CHUNKS * 32, tag, CHUNKS
                        )
        stamp(Trace, pid, 1)
        if not POOL:
            task = pid - 640
            if task >= 0 and task < TASKS:
                ready = wait_line(
                    State + p[8] + task * 32, tag, pid, task, Diag, STAGE, Record, 21
                )
                ok = ok & ready
                if ready:
                    row = task // CHUNKS
                    i = (task % CHUNKS) * CHUNK + gl.arange(
                        0, CHUNK, layout=gl.BlockedLayout([1], [32], [4], [0])
                    )
                    v = load_tile_words(
                        Updated.to(gl.pointer_type(gl.int32)) + row * WIDTH + i,
                        i >= 0,
                        CHUNK,
                    ).to(gl.float32, bitcast=True)
                    inv = scoped.load_flag(State + p[4] + row * 32).to(
                        gl.float32, bitcast=True
                    )
                    store_bf16_vector(
                        Out + row * WIDTH + (task % CHUNKS) * CHUNK,
                        v * inv * gl.load(Pre + i).to(gl.float32),
                        CHUNK,
                    )
                    if flow.arrive(State + p[10] + row * 32, tag, CHUNKS):
                        scoped.publish(State + p[9] + row * 32, tag)
    stamp(Trace, pid, 2)
    # Only one forwarding CTA polls each row flag; GEMM consumers keep private lines.
    if pid < 8:
        ready = wait_line(
            State + p[9] + pid * 32, tag, pid, pid, Diag, STAGE, Record, 22
        )
        ok = ok & ready
        if ready and Counter is not None:
            if flow.arrive(Counter, tag, 8):
                if POOL:
                    if flow.arrive(Join, tag, 2):
                        scoped.publish_flags(Flags, tag, NEXT)
                else:
                    scoped.publish_flags(Flags, tag, NEXT)
    stamp(Trace, pid, 3)
    return ok

"""S3 boundary, MI355-style column split (new stage design, numerically validated).

Replaces tlx_attn_tail.s3_boundary (8 CTAs, one full 8192-wide row each, three
dependent passes) by two short kernels of ROWS x S tasks:

  A  per (row, slice of 8192/S columns): one-shot all-reduce of the slice
     (per-element rank order identical to s3_boundary, so b is bit-identical),
     then four partial sums  sum b^2, sum (beta b)^2, sum (alpha h)(beta b),
     sum (alpha h)^2.
  B  per (row, slice): both norm scales from the S partial sums (one exchange;
     the pre-FFN sum of squares by expansion, which ignores the BF16 rounding
     of the normed branch), then mid_stream and x4 for the slice.
"""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard,
    tlx_standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_memory as mem,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_asm import (
    _load_asm,
    _store_asm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
    ROW4,
)

W = 8192


@triton.jit
def _drain_zero(x):
    """Wait for this wave's stores to land; returns 0 (orders later ops by data)."""
    return tl.inline_asm_elementwise(
        "s_wait_storecnt 0x0\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        constraints="=v,v",
        args=[x],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _signal_dev(ptr):
    """All lanes: one SCOPE_DEV atomic add of 1 each, after this wave's stores land."""
    return tl.inline_asm_elementwise(
        "s_wait_storecnt 0x0\nv_mov_b32 $0, 1\nglobal_atomic_add_u32 $1, $0, off scope:SCOPE_DEV\ns_wait_xcnt 0x0",
        constraints="=&v,v",
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def slice_ar_sums(
    Partial,
    Stream,
    Alpha,
    Beta,
    Reduced,
    Sums,
    Status,
    Epochs,
    Scratch,
    Peers,
    Failure,
    GRID: tl.constexpr,
    RANK: tl.constexpr,
    PAYLOAD: tl.constexpr,
    FLAGS: tl.constexpr,
    S: tl.constexpr = 8,
    Trace=None,
):
    WIDTH: tl.constexpr = 8192
    SW: tl.constexpr = WIDTH // S  # columns of the slice
    t_start = tlx_standin.clock()
    Q: tl.constexpr = SW // 8  # 4-word groups of the slice
    task = tl.program_id(0)
    row = task // S
    sl = task % S
    tag = tlx_standin.advance(Epochs, task)
    i512 = tlx.require_layout(tl.arange(0, 512), ROW1)
    ok = tl.min(tl.load(Status + i512), 0) == 1
    t_status = tlx_standin.clock()
    q = tlx.require_layout(tl.arange(0, Q), ROW1)
    at = row * (WIDTH // 2) + sl * (SW // 2) + q * 4
    v = mem.load_contiguous_words(
        Partial.to(tl.pointer_type(tl.int32))
        + RANK * 4 * WIDTH
        + row * (WIDTH // 2)
        + sl * (SW // 2),
        SW // 2,
        ROW4,
    )
    a, b, c, d = mem.split4(v, Q)
    pointers = ()
    for dst in tl.static_range(4):
        peer = tl.load(Peers + dst).to(tl.pointer_type(tl.int32))
        pointers += (
            peer
            + PAYLOAD[0]
            + dst * PAYLOAD[1]
            + (tag % 2) * PAYLOAD[2]
            + RANK * 4 * WIDTH
            + at,
        )
    spec: tl.constexpr = _store_asm(4, 1, True)
    mask = tlx.require_layout(tl.full((Q,), 1, tl.int32), ROW1)
    tl.inline_asm_elementwise(
        spec[0],
        spec[1],
        [a, b, c, d, pointers[0], pointers[1], pointers[2], pointers[3], mask],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    mem.drain_stores()
    t_push = tlx_standin.clock()
    i = tlx.require_layout(tl.arange(0, 128), ROW1)
    dstmask = i < 4
    peer = tl.load(Peers + i, dstmask, 0).to(tl.pointer_type(tl.int32))
    offset = FLAGS[0] + i * FLAGS[1] + (tag % 2) * FLAGS[2] + (RANK * GRID + task) * 32
    mem.store_words(peer + offset, tl.full((128,), tag, tl.int32), dstmask)
    ptr = Scratch + FLAGS[0] + RANK * FLAGS[1] + (tag % 2) * FLAGS[2] + (i * GRID + task) * 32
    seen = mem.load_words_masked(ptr, dstmask)
    ready = tl.sum(((seen == tag) & dstmask).to(tl.int32), 0) == 4
    spins = 1
    while (~ready) & (spins < 4096):
        seen = mem.load_words_masked(ptr, dstmask)
        ready = tl.sum(((seen == tag) & dstmask).to(tl.int32), 0) == 4
        spins += 1
    mem.drain_stores()
    t_poll = tlx_standin.clock()
    base = Scratch + PAYLOAD[0] + RANK * PAYLOAD[1] + (tag % 2) * PAYLOAD[2] + at
    spec2: tl.constexpr = _load_asm(4, 1)
    raw = tl.inline_asm_elementwise(
        spec2[0],
        spec2[1],
        [
            tlx.release_layout(base),
            tlx.release_layout(base + 4 * WIDTH),
            tlx.release_layout(base + 8 * WIDTH),
            tlx.release_layout(base + 12 * WIDTH),
            tlx.release_layout(mask),
        ],
        dtype=(tl.int32,) * 17,
        is_pure=False,
        pack=1,
    )
    z = tlx.require_layout(tl.full((Q,), 0, tl.float32), ROW1)
    a, b, c, d, e, f, g, h = z, z, z, z, z, z, z, z
    for src in tl.static_range(4):
        w0 = tlx.require_layout(raw[src * 4], ROW1)
        w1 = tlx.require_layout(raw[src * 4 + 1], ROW1)
        w2 = tlx.require_layout(raw[src * 4 + 2], ROW1)
        w3 = tlx.require_layout(raw[src * 4 + 3], ROW1)
        a, b = tlx_standin.pair(a, b, w0)
        c, d = tlx_standin.pair(c, d, w1)
        e, f = tlx_standin.pair(e, f, w2)
        g, h = tlx_standin.pair(g, h, w3)
    value = tlx.require_layout(mem.join8(a, b, c, d, e, f, g, h, Q), ROW1)
    t_load = tlx_standin.clock()
    ok = ok & ready
    j = tlx.require_layout(tl.arange(0, SW), ROW1)
    col = sl * SW + j
    bb = value.to(tl.bfloat16).to(tl.float32)
    hh = tl.load(Stream + row * WIDTH + col)
    al = tl.load(Alpha + col)
    be = tl.load(Beta + col)
    ah = al * hh
    bt = be * bb
    tl.store(Reduced + row * WIDTH + col, bb)
    k4 = tl.arange(0, 4)
    sums = tl.where(
        k4 == 0,
        tl.sum(bb * bb, 0),
        tl.where(
            k4 == 1,
            tl.sum(bt * bt, 0),
            tl.where(k4 == 2, tl.sum(ah * bt, 0), tl.sum(ah * ah, 0)),
        ),
    )
    tl.store(Sums + task * 4 + k4, sums)
    tl.store(Status + task, ok)
    if Trace is not None:
        t_end = tlx_standin.clock()
        k8 = tl.arange(0, 8)
        stamps = tl.where(k8 == 0, t_start, tl.where(k8 == 1, t_status, tl.where(k8 == 2, t_push,
                 tl.where(k8 == 3, t_poll, tl.where(k8 == 4, t_load, tl.where(k8 == 5, t_end, 0))))))
        tl.store(Trace + task * 8 + k8, stamps.to(tl.int64))
    tlx_guard.finish(Failure, ok, task, 3)


@triton.jit
def slice_fused(
    Partial,
    Stream,
    Alpha,
    Beta,
    Reduced,
    Sums,
    Status,
    Epochs,
    Scratch,
    Peers,
    Failure,
    GRID: tl.constexpr,
    RANK: tl.constexpr,
    PAYLOAD: tl.constexpr,
    FLAGS: tl.constexpr,
    S: tl.constexpr = 8,
    Trace=None,
    RowCnt=None,
    Pre=None,
    Out=None,
    Updated=None,
):
    WIDTH: tl.constexpr = 8192
    SW: tl.constexpr = WIDTH // S  # columns of the slice
    t_start = tlx_standin.clock()
    Q: tl.constexpr = SW // 8  # 4-word groups of the slice
    task = tl.program_id(0)
    row = task // S
    sl = task % S
    tag = tlx_standin.advance(Epochs, task)
    i512 = tlx.require_layout(tl.arange(0, 512), ROW1)
    ok = tl.min(tl.load(Status + i512), 0) == 1
    t_status = tlx_standin.clock()
    q = tlx.require_layout(tl.arange(0, Q), ROW1)
    at = row * (WIDTH // 2) + sl * (SW // 2) + q * 4
    v = mem.load_contiguous_words(
        Partial.to(tl.pointer_type(tl.int32))
        + RANK * 4 * WIDTH
        + row * (WIDTH // 2)
        + sl * (SW // 2),
        SW // 2,
        ROW4,
    )
    a, b, c, d = mem.split4(v, Q)
    pointers = ()
    for dst in tl.static_range(4):
        peer = tl.load(Peers + dst).to(tl.pointer_type(tl.int32))
        pointers += (
            peer
            + PAYLOAD[0]
            + dst * PAYLOAD[1]
            + (tag % 2) * PAYLOAD[2]
            + RANK * 4 * WIDTH
            + at,
        )
    spec: tl.constexpr = _store_asm(4, 1, True)
    mask = tlx.require_layout(tl.full((Q,), 1, tl.int32), ROW1)
    tl.inline_asm_elementwise(
        spec[0],
        spec[1],
        [a, b, c, d, pointers[0], pointers[1], pointers[2], pointers[3], mask],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    mem.drain_stores()
    t_push = tlx_standin.clock()
    i = tlx.require_layout(tl.arange(0, 128), ROW1)
    dstmask = i < 4
    peer = tl.load(Peers + i, dstmask, 0).to(tl.pointer_type(tl.int32))
    offset = FLAGS[0] + i * FLAGS[1] + (tag % 2) * FLAGS[2] + (RANK * GRID + task) * 32
    mem.store_words(peer + offset, tl.full((128,), tag, tl.int32), dstmask)
    ptr = Scratch + FLAGS[0] + RANK * FLAGS[1] + (tag % 2) * FLAGS[2] + (i * GRID + task) * 32
    seen = mem.load_words_masked(ptr, dstmask)
    ready = tl.sum(((seen == tag) & dstmask).to(tl.int32), 0) == 4
    spins = 1
    while (~ready) & (spins < 4096):
        seen = mem.load_words_masked(ptr, dstmask)
        ready = tl.sum(((seen == tag) & dstmask).to(tl.int32), 0) == 4
        spins += 1
    mem.drain_stores()
    t_poll = tlx_standin.clock()
    base = Scratch + PAYLOAD[0] + RANK * PAYLOAD[1] + (tag % 2) * PAYLOAD[2] + at
    spec2: tl.constexpr = _load_asm(4, 1)
    raw = tl.inline_asm_elementwise(
        spec2[0],
        spec2[1],
        [
            tlx.release_layout(base),
            tlx.release_layout(base + 4 * WIDTH),
            tlx.release_layout(base + 8 * WIDTH),
            tlx.release_layout(base + 12 * WIDTH),
            tlx.release_layout(mask),
        ],
        dtype=(tl.int32,) * 17,
        is_pure=False,
        pack=1,
    )
    z = tlx.require_layout(tl.full((Q,), 0, tl.float32), ROW1)
    a, b, c, d, e, f, g, h = z, z, z, z, z, z, z, z
    for src in tl.static_range(4):
        w0 = tlx.require_layout(raw[src * 4], ROW1)
        w1 = tlx.require_layout(raw[src * 4 + 1], ROW1)
        w2 = tlx.require_layout(raw[src * 4 + 2], ROW1)
        w3 = tlx.require_layout(raw[src * 4 + 3], ROW1)
        a, b = tlx_standin.pair(a, b, w0)
        c, d = tlx_standin.pair(c, d, w1)
        e, f = tlx_standin.pair(e, f, w2)
        g, h = tlx_standin.pair(g, h, w3)
    value = tlx.require_layout(mem.join8(a, b, c, d, e, f, g, h, Q), ROW1)
    t_load = tlx_standin.clock()
    ok = ok & ready
    j = tlx.require_layout(tl.arange(0, SW), ROW1)
    col = sl * SW + j
    bb = value.to(tl.bfloat16).to(tl.float32)
    hh = tl.load(Stream + row * WIDTH + col)
    al = tl.load(Alpha + col)
    be = tl.load(Beta + col)
    ah = al * hh
    bt = be * bb
    k4 = tl.arange(0, 4)
    sums = tl.where(
        k4 == 0,
        tl.sum(bb * bb, 0),
        tl.where(
            k4 == 1,
            tl.sum(bt * bt, 0),
            tl.where(k4 == 2, tl.sum(ah * bt, 0), tl.sum(ah * ah, 0)),
        ),
    )
    # publish this slice's sums + status (SCOPE_DEV asm stores), then signal the
    # row from asm after s_wait_storecnt: ordered, no cache writeback.
    s0 = tl.sum(bb * bb, 0)
    s1 = tl.sum(bt * bt, 0)
    s2 = tl.sum(ah * bt, 0)
    s3 = tl.sum(ah * ah, 0)
    k4w = tl.arange(0, 128)
    f0 = (k4w == 0).to(tl.float32)
    f1 = (k4w == 1).to(tl.float32)
    f2 = (k4w == 2).to(tl.float32)
    f3 = (k4w == 3).to(tl.float32)
    sw = f0 * s0 + f1 * s1 + f2 * s2 + f3 * s3
    mem.store_words(Sums + task * 4 + k4w, sw.to(tl.int32, bitcast=True), k4w < 4)
    mem.store_words(Status + task + k4w * 0, k4w * 0 + ok.to(tl.int32), k4w == 0)
    t_pub0 = tlx_standin.clock()
    z = _drain_zero(k4w)
    tlx.workgroup_barrier()
    zz = tl.max(z, 0)
    tl.atomic_add(RowCnt + row * 32 + zz + k4w * 0, 1, mask=k4w == 0, sem="relaxed",
                  scope="gpu")
    keep = zz
    target = S * tag
    c = mem.load_flag(RowCnt + row * 32 + keep)
    n = 0
    while (c < target) & (n < 20000):
        c = mem.load_flag(RowCnt + row * 32 + keep)
        n += 1
    t_wait = tlx_standin.clock()
    dep = tl.minimum(c - target, 0)
    lane = tlx.require_layout(tl.arange(0, 128), ROW1)
    got = mem.load_words_masked(Sums + dep + row * S * 4 + lane, lane < S * 4)
    gst = mem.load_words_masked(Status + dep + row * S + lane, lane < S)
    rows_ok = tl.sum(((gst == 1) & (lane < S)).to(tl.int32), 0) == S
    ok = ok & (c >= target) & rows_ok
    gv = got.to(tl.float32, bitcast=True)
    t0 = tl.sum(tl.where((lane % 4 == 0) & (lane < S * 4), gv, 0.0), 0)
    t1 = tl.sum(tl.where((lane % 4 == 1) & (lane < S * 4), gv, 0.0), 0)
    t2 = tl.sum(tl.where((lane % 4 == 2) & (lane < S * 4), gv, 0.0), 0)
    t3 = tl.sum(tl.where((lane % 4 == 3) & (lane < S * 4), gv, 0.0), 0)
    inv = tl.rsqrt(t0 / WIDTH + 1e-8)
    ssh = tl.maximum(inv * inv * t1 + 2.0 * inv * t2 + t3, 0.0)
    inv2 = tl.rsqrt(ssh / WIDTH + 1e-5)
    bn = (bb * inv).to(tl.bfloat16).to(tl.float32)
    updated = ah + be * bn
    tl.store(Updated + row * WIDTH + col, updated)
    pre = updated * inv2
    pre *= tl.load(Pre + col).to(tl.float32)
    tl.store(Out + row * WIDTH + col, pre.to(tl.bfloat16))
    if Trace is not None:
        t_end = tlx_standin.clock()
        k8 = tl.arange(0, 8)
        stamps = tl.where(k8 == 0, t_start, tl.where(k8 == 1, t_status, tl.where(k8 == 2, t_push,
                 tl.where(k8 == 3, t_poll, tl.where(k8 == 4, t_load, tl.where(k8 == 5, t_pub0,
                 tl.where(k8 == 6, t_wait, t_end)))))))
        tl.store(Trace + task * 8 + k8, stamps.to(tl.int64))
    tlx_guard.finish(Failure, ok, task, 3)


@triton.jit
def slice_finish(
    Reduced,
    Sums,
    Stream,
    Alpha,
    Beta,
    Pre,
    Out,
    Updated,
    Status,
    Failure,
    S: tl.constexpr = 8,
):
    WIDTH: tl.constexpr = 8192
    SW: tl.constexpr = WIDTH // S
    task = tl.program_id(0)
    row = task // S
    sl = task % S
    t = tl.arange(0, S)
    ok = tl.min(tl.load(Status + row * S + t), 0) == 1
    t0 = tl.sum(tl.load(Sums + (row * S + t) * 4 + 0), 0)
    t1 = tl.sum(tl.load(Sums + (row * S + t) * 4 + 1), 0)
    t2 = tl.sum(tl.load(Sums + (row * S + t) * 4 + 2), 0)
    t3 = tl.sum(tl.load(Sums + (row * S + t) * 4 + 3), 0)
    inv = tl.rsqrt(t0 / WIDTH + 1e-8)
    ssh = tl.maximum(inv * inv * t1 + 2.0 * inv * t2 + t3, 0.0)
    inv2 = tl.rsqrt(ssh / WIDTH + 1e-5)
    j = tlx.require_layout(tl.arange(0, SW), ROW1)
    col = sl * SW + j
    bb = tl.load(Reduced + row * WIDTH + col)
    bn = (bb * inv).to(tl.bfloat16).to(tl.float32)
    hh = tl.load(Stream + row * WIDTH + col)
    updated = tl.load(Alpha + col) * hh + tl.load(Beta + col) * bn
    tl.store(Updated + row * WIDTH + col, updated)
    pre = updated * inv2
    pre *= tl.load(Pre + col).to(tl.float32)
    tl.store(Out + row * WIDTH + col, pre.to(tl.bfloat16))
    tlx_guard.finish(Failure, ok, task, 3)


def boundary(partial, stream, pre, alpha, beta, out, updated, ws, *, rank, failure, sums, s=8, trace=None):
    """Two-kernel S3 boundary over 8 x s tasks (replaces s3_boundary[(8,)])."""
    p, st = ws.peers, ws.state
    slice_ar_sums[(8 * s,)](
        partial,
        stream,
        alpha,
        beta,
        p.reduced,
        sums,
        st.status,
        st.epochs,
        p.scratch,
        p.peers,
        failure,
        GRID=p.ctas,
        RANK=rank,
        PAYLOAD=p.payload,
        FLAGS=p.flags,
        S=s,
        Trace=trace,
        num_warps=4,
        enable_fp_fusion=False,
    )
    slice_finish[(8 * s,)](
        p.reduced,
        sums,
        stream,
        alpha,
        beta,
        pre,
        out,
        updated,
        st.status,
        failure,
        S=s,
        num_warps=4,
        enable_fp_fusion=False,
    )


def boundary_one(partial, stream, pre, alpha, beta, out, updated, ws, *, rank, failure, sums,
                 rowcnt, s=8, trace=None):
    """One-kernel S3 boundary over 8 x s tasks with an in-kernel per-row sums exchange."""
    p, st = ws.peers, ws.state
    slice_fused[(8 * s,)](
        partial, stream, alpha, beta, p.reduced, sums, st.status, st.epochs, p.scratch,
        p.peers, failure, GRID=p.ctas, RANK=rank, PAYLOAD=p.payload, FLAGS=p.flags, S=s,
        RowCnt=rowcnt, Pre=pre, Out=out, Updated=updated, Trace=trace, num_warps=4,
        enable_fp_fusion=False,
    )

"""TLX S4: exact FP32 SK8 router, pool norm and stable local slot table.

Dispatch-separated dependencies; no peer memory or polled shared flags.
Compute inputs preserve carriers; output stores/indices are compiler managed.
"""

import msgspec
import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard,
    tlx_workspace,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
    BlockedEncoding,
    require_pin_support,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_code_warm
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_projection
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_attention import (
    Slice,
    fixed,
)

# Each wave owns all 256 router logits, avoiding cross-wave top-k reductions.
# Pool normalization retains ROW1 and its original floating-point tree.
TOP = tl.constexpr(BlockedEncoding([8], [32], [4], [0]))
# top8_ids: two rows per wave, a row's 256 logits inside one wave.
TOP8 = tl.constexpr(BlockedEncoding([1, 8], [1, 32], [4, 1], [1, 0]))
IDS8 = tl.constexpr(BlockedEncoding([1, 8], [1, 32], [4, 1], [1, 0]))


@triton.jit
def finish_rows(P, Bias, Norm, Scores, Ids, Y, Producer, Rows, Failure, Seed=None):
    row = tl.program_id(0)
    check = tl.arange(0, 512)
    ok = tl.min(tl.load(Producer + check, check < 288, 1), 0) == 1
    tlx_guard.delay(Seed, row, 1)
    if ok:
        n = tl.arange(0, 256)
        high = fixed(
            tl.load(P + row * 4608 + 4096 + n).to(tl.float32, bitcast=True), TOP
        )
        low = fixed(
            tl.load(P + row * 4608 + 4352 + n).to(tl.float32, bitcast=True), TOP
        )
        raw = 1.0 / (1.0 + tl.exp(-(high + low) * 0.17677669529663687))
        biased = raw + fixed(tl.load(Bias + n), TOP)
        slots = tl.arange(0, 8)
        selected = fixed(tl.zeros((8,), tl.float32), TOP)
        ids = fixed(tl.zeros((8,), tl.int32), TOP)
        total = tl.sum(selected, 0)
        for j in tl.static_range(8):
            maximum = tl.max(biased, 0)
            index = tl.min(tl.where(biased == maximum, n, 2147483647), 0)
            value = tl.sum(tl.where(n == index, raw, 0.0), 0)
            total += value
            ids = tl.where(slots == j, index, ids)
            selected = tl.where(slots == j, value, selected)
            biased = tl.where(n == index, -float("inf"), biased)
        tl.store(Ids + row * 8 + slots, ids)
        tl.store(Scores + row * 8 + slots, selected / (total + 1e-15))
        d = tl.arange(0, 4096)
        pooled = fixed(
            tl.load(P + row * 4608 + d)
            .to(tl.float32, bitcast=True)
            .to(tl.bfloat16)
            .to(tl.float32),
            ROW1,
        )
        squares = fixed(pooled * pooled, ROW1)
        pooled *= tl.rsqrt(tl.sum(squares, 0) / 4096 + 1e-5)
        pooled *= fixed(tl.load(Norm + d).to(tl.float32), ROW1)
        tl.store(Y + row * 4096 + d, pooled.to(tl.bfloat16))
    tl.store(Rows + row, ok)
    tlx_guard.finish(Failure, ok, row, 4)


@triton.jit
def finish_row_body(row, P, Bias, Norm, Scores, Ids, Y, Producer, Rows, Failure, dep, ok_in,
                    EARLY: tl.constexpr = False):
    """finish_rows for one row as a device function (identical math/layouts);
    ``dep`` (0 at runtime) orders the P/Producer reads after the caller's wait.
    EARLY: P/Bias/Norm loads are issued before the producer-status read."""
    if EARLY:
        ne = tl.arange(0, 256)
        de = tl.arange(0, 4096)
        high0 = tl.load(P + dep + row * 4608 + 4096 + ne)
        low0 = tl.load(P + dep + row * 4608 + 4352 + ne)
        bias0 = tl.load(Bias + ne)
        pooled0 = tl.load(P + dep + row * 4608 + de)
        norm0 = tl.load(Norm + de)
    check = tl.arange(0, 512)
    ok = ok_in & (tl.min(tl.load(Producer + dep + check, check < 288, 1), 0) == 1)
    if ok:
        n = tl.arange(0, 256)
        if EARLY:
            high = fixed(high0.to(tl.float32, bitcast=True), TOP)
            low = fixed(low0.to(tl.float32, bitcast=True), TOP)
            bias = bias0
        else:
            high = fixed(
                tl.load(P + dep + row * 4608 + 4096 + n).to(tl.float32, bitcast=True), TOP
            )
            low = fixed(
                tl.load(P + dep + row * 4608 + 4352 + n).to(tl.float32, bitcast=True), TOP
            )
            bias = tl.load(Bias + n)
        raw = 1.0 / (1.0 + tl.exp(-(high + low) * 0.17677669529663687))
        biased = raw + fixed(bias, TOP)
        slots_ = tl.arange(0, 8)
        selected = fixed(tl.zeros((8,), tl.float32), TOP)
        ids = fixed(tl.zeros((8,), tl.int32), TOP)
        total = tl.sum(selected, 0)
        for j in tl.static_range(8):
            maximum = tl.max(biased, 0)
            index = tl.min(tl.where(biased == maximum, n, 2147483647), 0)
            value = tl.sum(tl.where(n == index, raw, 0.0), 0)
            total += value
            ids = tl.where(slots_ == j, index, ids)
            selected = tl.where(slots_ == j, value, selected)
            biased = tl.where(n == index, -float("inf"), biased)
        tl.store(Ids + row * 8 + slots_, ids)
        tl.store(Scores + row * 8 + slots_, selected / (total + 1e-15))
        d = tl.arange(0, 4096)
        pooled = fixed(
            (pooled0 if EARLY else tl.load(P + dep + row * 4608 + d))
            .to(tl.float32, bitcast=True)
            .to(tl.bfloat16)
            .to(tl.float32),
            ROW1,
        )
        squares = fixed(pooled * pooled, ROW1)
        pooled *= tl.rsqrt(tl.sum(squares, 0) / 4096 + 1e-5)
        pooled *= fixed((norm0 if EARLY else tl.load(Norm + d)).to(tl.float32), ROW1)
        tl.store(Y + row * 4096 + d, pooled.to(tl.bfloat16))
    tl.store(Rows + row, ok)
    tlx_guard.finish(Failure, ok, row, 4)


@triton.jit(do_not_specialize=[
    "L0", "H0", "L1", "H1", "L2", "H2", "L3", "H3", "L4", "H4", "L5", "H5", "L6", "H6", "L7",
    "H7", "L8", "H8", "L9", "H9", "L10", "H10", "L11", "H11", "B1", "B2"])
def slots(
    Ids,
    Sorted,
    Experts,
    NumPost,
    Rows,
    Status,
    Failure,
    LOCAL_START: tl.constexpr,
    GRID: tl.constexpr,
    Seed=None,
    Mine=None,
    L0=0, H0=0, L1=0, H1=0, L2=0, H2=0, L3=0, H3=0, L4=0, H4=0, L5=0, H5=0, L6=0, H6=0, L7=0,
    H7=0, L8=0, H8=0, L9=0, H9=0, L10=0, H10=0, L11=0, H11=0, B1=0, B2=0,
):
    pid = tl.program_id(0)
    ok = tl.min(tl.load(Rows + tl.arange(0, 8)), 0) == 1
    tlx_guard.delay(Seed, pid, 2)
    ids = tl.zeros((64,), tl.int32)
    if ok:
        ids = tl.load(Ids + tl.arange(0, 64))
    slots_body(pid, ids, ok, Sorted, Experts, NumPost, Status, Failure, LOCAL_START, GRID, Mine,
               L0, H0, L1, H1, L2, H2, L3, H3, L4, H4, L5, H5, L6, H6, L7, H7, L8, H8, L9, H9,
               L10, H10, L11, H11, B1, B2)


@triton.jit
def top8_ids(P, Bias, Pre=None):
    """The 8 rows' routed expert ids, [64] = row * 8 + slot. Max/min selection
    only, so the ids are the bits finish_rows stores whatever the layout.
    Pre = (high, low, bias) already loaded by the caller."""
    r = tl.arange(0, 8)[:, None]
    n = fixed(tl.broadcast_to(tl.arange(0, 256)[None, :], (8, 256)), TOP8)
    if Pre is not None:
        high = fixed(Pre[0].to(tl.float32, bitcast=True), TOP8)
        low = fixed(Pre[1].to(tl.float32, bitcast=True), TOP8)
        bias = Pre[2]
    else:
        high = fixed(tl.load(P + r * 4608 + 4096 + n).to(tl.float32, bitcast=True), TOP8)
        low = fixed(tl.load(P + r * 4608 + 4352 + n).to(tl.float32, bitcast=True), TOP8)
        bias = tl.load(Bias + n)
    raw = 1.0 / (1.0 + tl.exp(-(high + low) * 0.17677669529663687))
    biased = fixed(raw + bias, TOP8)
    sl = fixed(tl.broadcast_to(tl.arange(0, 8)[None, :], (8, 8)), IDS8)
    ids = fixed(tl.zeros((8, 8), tl.int32), IDS8)
    for j in tl.static_range(8):
        maximum = tl.max(biased, 1)
        index = tl.min(tl.where(biased == maximum[:, None], n, 2147483647), 1)
        ids = tl.where(sl == j, fixed(tl.broadcast_to(index[:, None], (8, 8)), IDS8), ids)
        biased = tl.where(n == index[:, None], -float("inf"), biased)
    return tlx.release_layout(tl.reshape(tlx.release_layout(ids), (64,)))


# rows_slots issues its S4-output/bias/norm loads before the producer-status read
EARLY_RS = True

QPART = tl.constexpr(BlockedEncoding([1, 4], [4, 8], [4, 1], [1, 0]))


@triton.jit
def mx_term(x):
    """one MXFP8 term of x [16, 32]: (e4m3 values, e8m0 scales, dequantized value)"""
    amax = tl.max(tl.abs(x), 1)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    se = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    se = tl.minimum(se, 253)
    inv8 = ((254 - se) << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x * inv8[:, None], -448.0), 448.0).to(tl.float8e4nv)
    # (se << 23) as fp32 is exactly 2^(se - 127), the e8m0 scale
    back = q.to(tl.float32) * (se << 23).to(tl.float32, bitcast=True)[:, None]
    return q, se, back


@triton.jit
def quant_part(k, P, Norm, YQ, YS, Producer, YQ2=None, YS2=None):
    """MXFP8 of x4 row k // 8, columns (k % 8) * 512 .. +512 (W13's activations).
    The row's rms factor is recomputed with finish_row_body's exact ROW1 tree, so the
    quantized values are those of the stored bf16 Y."""
    row = k // 8
    part = k % 8
    check = tl.arange(0, 512)
    ok = tl.min(tl.load(Producer + check, check < 288, 1), 0) == 1
    if ok:
        d = tl.arange(0, 4096)
        pooled = fixed(
            tl.load(P + row * 4608 + d).to(tl.float32, bitcast=True).to(tl.bfloat16)
            .to(tl.float32),
            ROW1,
        )
        squares = fixed(pooled * pooled, ROW1)
        inv = tl.rsqrt(tl.sum(squares, 0) / 4096 + 1e-5)
        b = tl.arange(0, 16)
        j = tl.arange(0, 32)
        col = tlx.require_layout(part * 512 + b[:, None] * 32 + j[None, :], QPART)
        x = tl.load(P + row * 4608 + col).to(tl.float32, bitcast=True).to(tl.bfloat16)
        x = x.to(tl.float32) * inv
        x = (x * tl.load(Norm + col).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        q, se, back = mx_term(x)
        tl.store(YQ + row * 4096 + col, q.to(tl.uint8, bitcast=True))
        tl.store(YS + row * 128 + part * 16 + b, se.to(tl.uint8))
        if YQ2 is not None:
            # second term: the exact fp32 residual (bf16 x = hi + lo exactly, almost always)
            q2, se2, _ = mx_term(x - back)
            tl.store(YQ2 + row * 4096 + col, q2.to(tl.uint8, bitcast=True))
            tl.store(YS2 + row * 128 + part * 16 + b, se2.to(tl.uint8))


# probe (MK.50): rows_slots reads its own .text into every XCD's L2 at launch start
CODE_WARM_RS = False
_SINK = {}  # device -> int32 [512] sink (written only when quads < 0, i.e. never); allocate outside capture


def _code_operands(fn, args, kwargs, grid, device):
    """Code offset / size are runtime ints that are never specialized (do_not_specialize), so
    the binary does not depend on their values and no host->device copy is needed (graph
    capture safe)."""
    sink = _SINK.get(str(device))
    if sink is None:
        sink = _SINK[str(device)] = torch.zeros(512, device=device, dtype=torch.int32)
    common = dict(Sink=sink, NCTA=grid, WARM=True)
    kernel = fn.warmup(*args, **kwargs, **common, CodeOff=0, CodeQuads=0, grid=(grid,))
    offset, quads = tlx_code_warm.text_range(kernel)
    return dict(**common, CodeOff=offset, CodeQuads=quads)


@triton.jit(do_not_specialize=["CodeOff", "CodeQuads", 
    "L0", "H0", "L1", "H1", "L2", "H2", "L3", "H3", "L4", "H4", "L5", "H5", "L6", "H6", "L7",
    "H7", "L8", "H8", "L9", "H9", "L10", "H10", "L11", "H11", "B1", "B2"])
def rows_slots(P, Bias, Norm, Scores, Ids, Y, Producer, Rows, Sorted, Experts, NumPost,
               Status, Failure, LOCAL_START: tl.constexpr, GRID: tl.constexpr, Seed=None,
               EARLY: tl.constexpr = False, YQ=None, YS=None, YQ2=None, YS2=None,
               Sink=None, NCTA: tl.constexpr = 72, WARM: tl.constexpr = False,
               CodeOff=0, CodeQuads=0, Mine=None,
               L0=0, H0=0, L1=0, H1=0, L2=0, H2=0, L3=0, H3=0, L4=0, H4=0, L5=0, H5=0, L6=0,
               H6=0, L7=0, H7=0, L8=0, H8=0, L9=0, H9=0, L10=0, H10=0, L11=0, H11=0, B1=0,
               B2=0):
    """finish_rows (CTAs 0-7) and slots (CTAs 8-71) in one launch: each slot CTA
    recomputes the 64 ids itself instead of waiting for the row CTAs.
    EARLY: the S4 outputs/bias/norm loads go out before the producer-status read
    (all complete at launch; values and math unchanged)."""
    pid = tl.program_id(0)
    if WARM:
        tlx_code_warm.code_warm(pid, NCTA, Sink, CodeOff, CodeQuads)
    if pid < 8:
        finish_row_body(pid, P, Bias, Norm, Scores, Ids, Y, Producer, Rows, Failure, 0, True,
                        EARLY)
    elif pid >= 72:
        if YQ is not None:
            quant_part(pid - 72, P, Norm, YQ, YS, Producer, YQ2, YS2)
    else:
        pre = None
        if EARLY:
            r0 = tl.arange(0, 8)[:, None]
            n0 = tl.broadcast_to(tl.arange(0, 256)[None, :], (8, 256))
            pre = (tl.load(P + r0 * 4608 + 4096 + n0), tl.load(P + r0 * 4608 + 4352 + n0),
                   tl.load(Bias + n0))
        check = tl.arange(0, 512)
        ok = tl.min(tl.load(Producer + check, check < 288, 1), 0) == 1
        tlx_guard.delay(Seed, pid - 8, 2)
        ids = tl.zeros((64,), tl.int32)
        if ok:
            ids = top8_ids(P, Bias, pre)
        slots_body(pid - 8, ids, ok, Sorted, Experts, NumPost, Status, Failure, LOCAL_START, GRID,
                   Mine, 
                       L0, H0, L1, H1, L2, H2, L3, H3, L4, H4, L5, H5, L6, H6, L7, H7, L8, H8,
                       L9, H9, L10, H10, L11, H11, B1, B2)


# Experiment (MK.91-94): some experts have a second copy (copy B) on another rank; each step the
# routed experts are assigned to one of their ranks by a deterministic rule that every rank
# evaluates on the same 64 ids, so the busiest rank's work drops towards the mean.
#
# Placement is static and table-driven (one table per layer, the same on every rank):
#   mask[a][k] (uint64 over local expert ids, a = home rank e // 64, k = 0 / 1 / 2): the experts
#   of rank a whose copy B is on rank (a + 1 + k) % 4. Masks of one rank are disjoint; an
#   expert in no mask has no copy. On its second rank an expert has local index
#   64 + base[t][k] + (its position among the set bits of its mask), i.e. the B weights of rank
#   t are stored compactly: group k = 0 (from rank t - 1), then k = 1 (t - 2), then k = 2 (t - 3).
# Rule (see repl_local): per rank, load = distinct routed experts with that home; per (a, k)
# group, movable = routed experts in mask[a][k]; three passes of signed pairwise levelling over
# the 12 groups (shift half the load difference between a and the group's second rank, in
# either direction, within 0 <= moved <= movable) give per-group move counts; the lowest
# routed bits of a group move. Within one block of the exact min-max assignment in every case
# tried (mk/repl_gap.py).
# Reference implementation and test: test/manual/kernels/muse_spark_mono/mk/repl_slots.py.
REPL = False
REPL_TABLE = None  # masks[a][k] as Python ints (None = repl_masks_full()); see repl_operands()
_MINE = {}  # sorted_ids.data_ptr() -> int32 [64]: 1 where the route runs on this rank


def repl_masks_full():
    """Every expert has a copy: local ids 0-21 / 22-42 / 43-63 go to rank a + 1 / a + 2 / a + 3."""
    ranges = ((1 << 22) - 1, ((1 << 21) - 1) << 22, ((1 << 21) - 1) << 43)
    return [list(ranges) for _ in range(4)]


def repl_bases(masks):
    """base[t][k]: first compact B slot on rank t of the group that arrives with offset k."""
    out = []
    for t in range(4):
        n0 = bin(masks[(t - 1) % 4][0]).count("1")
        n1 = bin(masks[(t - 2) % 4][1]).count("1")
        n2 = bin(masks[(t - 3) % 4][2]).count("1")
        out.append([0, n0, n0 + n1, n0 + n1 + n2])  # [3] = number of B experts on rank t
    return out


def repl_operands(rank, masks=None):
    """Kernel scalar operands of the placement table for one rank: 12 masks (two 32-bit halves
    each) and this rank's compact-slot bases. Runtime ints (do_not_specialize), captured by
    value in a CUDA graph; the table of a layer never changes."""
    masks = (REPL_TABLE if REPL_TABLE is not None else repl_masks_full()) if masks is None else masks
    for a in range(4):
        assert masks[a][0] & masks[a][1] == 0 and masks[a][0] & masks[a][2] == 0
        assert masks[a][1] & masks[a][2] == 0 and all(0 <= m < 1 << 64 for m in masks[a])
    bases = repl_bases(masks)
    # two signed 32-bit halves per mask: a Python int operand is typed by its value, and a
    # mask that fits 32 bits would otherwise select a different binary than one that does not
    half = lambda v: v - (1 << 32) if v >= 1 << 31 else v  # noqa: E731
    out = {}
    for a in range(4):
        for k in range(3):
            out[f"L{3 * a + k}"] = half(masks[a][k] & 0xFFFFFFFF)
            out[f"H{3 * a + k}"] = half(masks[a][k] >> 32)
    out.update(B1=bases[rank][1], B2=bases[rank][2])
    return out


def repl_mine(sorted_ids):
    key = sorted_ids.data_ptr()
    if key not in _MINE:
        _MINE[key] = torch.zeros(64, device=sorted_ids.device, dtype=torch.int32)
    return _MINE[key]


@triton.jit
def _bit_or(x, y):
    return x | y


@triton.jit
def _popcount64(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((x * 0x0101010101010101) >> 56).to(tl.int32)


@triton.jit
def repl_local(ids, RANK: tl.constexpr, 
    L0, H0, L1, H1, L2, H2, L3, H3, L4, H4, L5, H5, L6, H6, L7, H7, L8, H8, L9, H9, L10, H10,
    L11, H11, B1, B2):
    """Per route: local expert index on this rank (0..63 own, 64.. second copies, -1 if
    another rank runs it), its block index, and this rank's block count."""
    # An id outside 0..255 is not a route (a row whose router logits are not finite selects
    # index INT_MAX; that happens after a failed exchange). It must not reach the tables: with
    # a = id >> 6 outside 0..3 the selects below would mix one rank's masks with another
    # rank's slot bases and produce a local index past this rank's tensors (MK.101).
    valid = (ids >= 0) & (ids < 256)
    a = ids >> 6
    ep = ids & 63
    ep64 = ep.to(tl.uint64)
    one = (ep64 * 0) + 1
    bit = tl.where(valid, one << ep64, one * 0)
    low = (one << ep64) - 1
    p0 = tl.reduce(tl.where(a == 0, bit, one * 0), 0, _bit_or)
    p1 = tl.reduce(tl.where(a == 1, bit, one * 0), 0, _bit_or)
    p2 = tl.reduce(tl.where(a == 2, bit, one * 0), 0, _bit_or)
    p3 = tl.reduce(tl.where(a == 3, bit, one * 0), 0, _bit_or)
    z = p0 * 0
    t0 = ((z + H0) << 32) | ((z + L0) & 0xFFFFFFFF)
    t1 = ((z + H1) << 32) | ((z + L1) & 0xFFFFFFFF)
    t2 = ((z + H2) << 32) | ((z + L2) & 0xFFFFFFFF)
    t3 = ((z + H3) << 32) | ((z + L3) & 0xFFFFFFFF)
    t4 = ((z + H4) << 32) | ((z + L4) & 0xFFFFFFFF)
    t5 = ((z + H5) << 32) | ((z + L5) & 0xFFFFFFFF)
    t6 = ((z + H6) << 32) | ((z + L6) & 0xFFFFFFFF)
    t7 = ((z + H7) << 32) | ((z + L7) & 0xFFFFFFFF)
    t8 = ((z + H8) << 32) | ((z + L8) & 0xFFFFFFFF)
    t9 = ((z + H9) << 32) | ((z + L9) & 0xFFFFFFFF)
    t10 = ((z + H10) << 32) | ((z + L10) & 0xFFFFFFFF)
    t11 = ((z + H11) << 32) | ((z + L11) & 0xFFFFFFFF)
    g0 = _popcount64(p0 & t0)
    m0 = g0 * 0
    g1 = _popcount64(p0 & t1)
    m1 = g1 * 0
    g2 = _popcount64(p0 & t2)
    m2 = g2 * 0
    g3 = _popcount64(p1 & t3)
    m3 = g3 * 0
    g4 = _popcount64(p1 & t4)
    m4 = g4 * 0
    g5 = _popcount64(p1 & t5)
    m5 = g5 * 0
    g6 = _popcount64(p2 & t6)
    m6 = g6 * 0
    g7 = _popcount64(p2 & t7)
    m7 = g7 * 0
    g8 = _popcount64(p2 & t8)
    m8 = g8 * 0
    g9 = _popcount64(p3 & t9)
    m9 = g9 * 0
    g10 = _popcount64(p3 & t10)
    m10 = g10 * 0
    g11 = _popcount64(p3 & t11)
    m11 = g11 * 0
    l0 = _popcount64(p0)
    l1 = _popcount64(p1)
    l2 = _popcount64(p2)
    l3 = _popcount64(p3)
    if RANK == 0:
        own = l0
    if RANK == 1:
        own = l1
    if RANK == 2:
        own = l2
    if RANK == 3:
        own = l3
    d = l0 - l1
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g0 - m0, h), -m0)
    m0 += mv
    l0 -= mv
    l1 += mv
    d = l0 - l2
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g1 - m1, h), -m1)
    m1 += mv
    l0 -= mv
    l2 += mv
    d = l0 - l3
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g2 - m2, h), -m2)
    m2 += mv
    l0 -= mv
    l3 += mv
    d = l1 - l2
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g3 - m3, h), -m3)
    m3 += mv
    l1 -= mv
    l2 += mv
    d = l1 - l3
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g4 - m4, h), -m4)
    m4 += mv
    l1 -= mv
    l3 += mv
    d = l1 - l0
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g5 - m5, h), -m5)
    m5 += mv
    l1 -= mv
    l0 += mv
    d = l2 - l3
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g6 - m6, h), -m6)
    m6 += mv
    l2 -= mv
    l3 += mv
    d = l2 - l0
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g7 - m7, h), -m7)
    m7 += mv
    l2 -= mv
    l0 += mv
    d = l2 - l1
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g8 - m8, h), -m8)
    m8 += mv
    l2 -= mv
    l1 += mv
    d = l3 - l0
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g9 - m9, h), -m9)
    m9 += mv
    l3 -= mv
    l0 += mv
    d = l3 - l1
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g10 - m10, h), -m10)
    m10 += mv
    l3 -= mv
    l1 += mv
    d = l3 - l2
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g11 - m11, h), -m11)
    m11 += mv
    l3 -= mv
    l2 += mv
    d = l0 - l1
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g0 - m0, h), -m0)
    m0 += mv
    l0 -= mv
    l1 += mv
    d = l0 - l2
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g1 - m1, h), -m1)
    m1 += mv
    l0 -= mv
    l2 += mv
    d = l0 - l3
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g2 - m2, h), -m2)
    m2 += mv
    l0 -= mv
    l3 += mv
    d = l1 - l2
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g3 - m3, h), -m3)
    m3 += mv
    l1 -= mv
    l2 += mv
    d = l1 - l3
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g4 - m4, h), -m4)
    m4 += mv
    l1 -= mv
    l3 += mv
    d = l1 - l0
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g5 - m5, h), -m5)
    m5 += mv
    l1 -= mv
    l0 += mv
    d = l2 - l3
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g6 - m6, h), -m6)
    m6 += mv
    l2 -= mv
    l3 += mv
    d = l2 - l0
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g7 - m7, h), -m7)
    m7 += mv
    l2 -= mv
    l0 += mv
    d = l2 - l1
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g8 - m8, h), -m8)
    m8 += mv
    l2 -= mv
    l1 += mv
    d = l3 - l0
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g9 - m9, h), -m9)
    m9 += mv
    l3 -= mv
    l0 += mv
    d = l3 - l1
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g10 - m10, h), -m10)
    m10 += mv
    l3 -= mv
    l1 += mv
    d = l3 - l2
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g11 - m11, h), -m11)
    m11 += mv
    l3 -= mv
    l2 += mv
    d = l0 - l1
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g0 - m0, h), -m0)
    m0 += mv
    l0 -= mv
    l1 += mv
    d = l0 - l2
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g1 - m1, h), -m1)
    m1 += mv
    l0 -= mv
    l2 += mv
    d = l0 - l3
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g2 - m2, h), -m2)
    m2 += mv
    l0 -= mv
    l3 += mv
    d = l1 - l2
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g3 - m3, h), -m3)
    m3 += mv
    l1 -= mv
    l2 += mv
    d = l1 - l3
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g4 - m4, h), -m4)
    m4 += mv
    l1 -= mv
    l3 += mv
    d = l1 - l0
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g5 - m5, h), -m5)
    m5 += mv
    l1 -= mv
    l0 += mv
    d = l2 - l3
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g6 - m6, h), -m6)
    m6 += mv
    l2 -= mv
    l3 += mv
    d = l2 - l0
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g7 - m7, h), -m7)
    m7 += mv
    l2 -= mv
    l0 += mv
    d = l2 - l1
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g8 - m8, h), -m8)
    m8 += mv
    l2 -= mv
    l1 += mv
    d = l3 - l0
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g9 - m9, h), -m9)
    m9 += mv
    l3 -= mv
    l0 += mv
    d = l3 - l1
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g10 - m10, h), -m10)
    m10 += mv
    l3 -= mv
    l1 += mv
    d = l3 - l2
    h = tl.where(d < 0, -((-d) >> 1), d >> 1)
    mv = tl.maximum(tl.minimum(g11 - m11, h), -m11)
    m11 += mv
    l3 -= mv
    l2 += mv
    # per route: its home bitmap, its group (3 = no copy) and that group's mask
    zv = ep64 * 0  # explicit vectors: a select between two scalars does not broadcast here
    zi = ep * 0
    pa = tl.where(a == 0, p0, tl.where(a == 1, p1, tl.where(a == 2, p2, zv + p3)))
    tm0 = tl.where(a == 0, t0, tl.where(a == 1, t3, tl.where(a == 2, t6, zv + t9)))
    tm1 = tl.where(a == 0, t1, tl.where(a == 1, t4, tl.where(a == 2, t7, zv + t10)))
    tm2 = tl.where(a == 0, t2, tl.where(a == 1, t5, tl.where(a == 2, t8, zv + t11)))
    k = tl.where((tm0 & bit) != 0, 0, tl.where((tm1 & bit) != 0, 1,
                 tl.where((tm2 & bit) != 0, 2, zi + 3)))
    tml = tl.where(k == 0, tm0, tl.where(k == 1, tm1, tm2))
    rk = _popcount64(pa & tml & low)
    g = a * 3 + k
    mvec = tl.zeros((64,), tl.int32)
    mvec = tl.where((g == 0) & (k < 3), m0, mvec)
    mvec = tl.where((g == 1) & (k < 3), m1, mvec)
    mvec = tl.where((g == 2) & (k < 3), m2, mvec)
    mvec = tl.where((g == 3) & (k < 3), m3, mvec)
    mvec = tl.where((g == 4) & (k < 3), m4, mvec)
    mvec = tl.where((g == 5) & (k < 3), m5, mvec)
    mvec = tl.where((g == 6) & (k < 3), m6, mvec)
    mvec = tl.where((g == 7) & (k < 3), m7, mvec)
    mvec = tl.where((g == 8) & (k < 3), m8, mvec)
    mvec = tl.where((g == 9) & (k < 3), m9, mvec)
    mvec = tl.where((g == 10) & (k < 3), m10, mvec)
    mvec = tl.where((g == 11) & (k < 3), m11, mvec)
    useb = valid & (k < 3) & (rk < mvec)
    b = (a + 1 + k) & 3
    mine = valid & (tl.where(useb, b, a) == RANK)
    if RANK == 0:
        r0 = t0
        r1 = t1
        r2 = t2
        mr0 = m0
        mr1 = m1
        mr2 = m2
        mb0 = m9
        mb1 = m7
        mb2 = m5
    if RANK == 1:
        r0 = t3
        r1 = t4
        r2 = t5
        mr0 = m3
        mr1 = m4
        mr2 = m5
        mb0 = m0
        mb1 = m10
        mb2 = m8
    if RANK == 2:
        r0 = t6
        r1 = t7
        r2 = t8
        mr0 = m6
        mr1 = m7
        mr2 = m8
        mb0 = m3
        mb1 = m1
        mb2 = m11
    if RANK == 3:
        r0 = t9
        r1 = t10
        r2 = t11
        mr0 = m9
        mr1 = m10
        mr2 = m11
        mb0 = m6
        mb1 = m4
        mb2 = m2
    base1 = (own * 0) + B1
    base2 = (own * 0) + B2
    slot = tl.where(k == 1, base1, tl.where(k == 2, base2, zi)) + _popcount64(tml & low)
    local = tl.where(useb, 64 + slot, ep)
    # blocks of this rank: its staying experts in bit order, then arriving ones by (k, bit)
    below = pa & low
    gone = (tl.minimum(_popcount64(below & r0), mr0) + tl.minimum(_popcount64(below & r1), mr1)
            + tl.minimum(_popcount64(below & r2), mr2))
    na = own - mr0 - mr1 - mr2
    blk_a = _popcount64(below) - gone
    blk_b = na + tl.where(k == 0, 0, tl.where(k == 1, mb0, zi + mb0 + mb1)) + rk
    blk = tl.where(useb, blk_b, blk_a)
    return tl.where(mine, local, -1), blk, na + mb0 + mb1 + mb2


@triton.jit
def slots_body(pid, ids, ok, Sorted, Experts, NumPost, Status, Failure,
               LOCAL_START: tl.constexpr, GRID: tl.constexpr, Mine=None,
               L0=0, H0=0, L1=0, H1=0, L2=0, H2=0, L3=0, H3=0, L4=0, H4=0, L5=0, H5=0, L6=0,
               H6=0, L7=0, H7=0, L8=0, H8=0, L9=0, H9=0, L10=0, H10=0, L11=0, H11=0, B1=0,
               B2=0):
    REPL_: tl.constexpr = Mine is not None
    if ok:
        if REPL_:
            # one block per distinct local expert (an expert has at most 8 routes)
            # one id per thread: the fused launch hands over ids in the top-8 layout (8 per
            # thread), which would unroll every operation below 8 times (MK.94)
            raw, blk, nblk = repl_local(fixed(ids, ROW1), LOCAL_START // 64,
                                        L0, H0, L1, H1, L2, H2, L3, H3, L4, H4, L5, H5, L6, H6,
                                        L7, H7, L8, H8, L9, H9, L10, H10, L11, H11, B1, B2)
            raw = tlx.release_layout(raw)
            blk = tlx.release_layout(blk)
            if pid == 0:
                tl.store(Mine + tl.arange(0, 64), (raw >= 0).to(tl.int32))
            match = (raw >= 0) & (blk == pid)
            owner = tl.max(tl.where(match, raw, -1), 0)
            owner = tl.where(owner >= 0, owner, 128)
            total = nblk * 16
            base = pid * 16
        else:
            j = tl.arange(0, 64)
            raw = ids - LOCAL_START
            counts = tl.histogram(raw, 64, mask=(raw >= 0) & (raw < 64))
            padded = tl.cdiv(counts, 16) * 16
            ends = tl.cumsum(padded, 0)
            total = tl.sum(padded, 0)
            owner = tl.min(tl.where(ends > pid * 16, j, 64), 0)
            base = tl.sum(tl.where(j < owner, padded, 0), 0)
            match = (raw == owner) & (owner < 64)
        rank = tl.cumsum(match.to(tl.int32), 0) - 1
        d = tl.arange(0, 16)
        nth = pid * 16 - base + d
        je = tl.arange(0, 64)
        hit = tl.min(
            tl.where(match[None, :] & (rank[None, :] == nth[:, None]), je[None, :], 64),
            1,
        )
        tl.store(Sorted + pid * 16 + d, hit)
        tl.store(Experts + pid, tl.where(owner < (128 if REPL_ else 64), owner, -1))
        if pid == 0:
            tl.store(NumPost, total)
    status = pid * 16 + tl.arange(0, 16)
    tl.store(Status + status, ok, status < GRID)
    tlx_guard.finish(Failure, ok, pid, 4)


class Workspace(tlx_workspace.S4Workspace):
    producer: object
    rows_ready: object
    local_failure: object


def make_workspace(device, *, shape=None):
    base = tlx_workspace.make_s4(device, shape=shape)
    return Workspace(
        *msgspec.structs.astuple(base),
        torch.empty(288, device=device, dtype=torch.int32),
        torch.empty(8, device=device, dtype=torch.int32),
        torch.zeros(2, device=device, dtype=torch.int32),
    )


def routing(
    x,
    weight,
    bias,
    norm,
    *,
    rank=0,
    workspace=None,
    delay_seed=None,
    skip=-1,
    failure=None,
    guard_mode=0,
    fused=False,
    mx=None,
):
    assert guard_mode == 0
    assert mx is None or fused, "MXFP8 x4 is produced by the fused rows_slots launch"
    require_pin_support()
    ws = make_workspace(x.device) if workspace is None else workspace
    failure = ws.local_failure if failure is None else failure
    pa = (x, weight, ws.projection, ws.producer)
    po = dict(
        N=4608,
        K=8192,
        SK=8,
        STAGE=4,
        ROUND_OUTPUT=False,
        Failure=failure,
        SKIP=skip,
        Seed=delay_seed,
        num_warps=2,
        enable_fp_fusion=False,
    )
    ra = (
        ws.projection,
        bias,
        norm,
        ws.scores,
        ws.ids,
        ws.pooled,
        ws.producer,
        ws.rows_ready,
        failure,
    )
    ro = dict(Seed=delay_seed, num_warps=4, enable_fp_fusion=False)
    sa = (
        ws.ids,
        ws.sorted_ids,
        ws.expert_ids,
        ws.num_post,
        ws.rows_ready,
        ws.state.status,
        failure,
    )
    so = dict(
        LOCAL_START=rank * 64,
        GRID=ws.state.grid,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if REPL:
        so["Mine"] = repl_mine(ws.sorted_ids)
        so.update(repl_operands(rank))
    key = (rank, norm.dtype, delay_seed is not None, skip, failure is not None, int(REPL))
    if key not in ws.state.admitted:
        records = []
        for phase, fn, args, opts, grid in [
            ("projection", tlx_projection.project, pa, po, 288),
            ("rows", finish_rows, ra, ro, 8),
            ("slots", slots, sa, so, 64),
        ]:
            kernel = fn.warmup(*args, **opts, grid=(grid,))
            r = kernel_occupancy("TLX-S4-" + phase, kernel)
            assert r.active_ctas_per_cu >= 1
            records.append(
                dict(
                    phase=phase,
                    vgprs=kernel.n_regs,
                    spills=kernel.n_spills,
                    lds=kernel.metadata.shared,
                    occupancy=r.active_ctas_per_cu,
                    grid=grid,
                )
            )
        ws.state.admitted[key] = records
    tlx_projection.project[(4608 // tlx_projection.BN_S4,)](*pa, **po, EARLY=tlx_projection.EARLY_ISSUE, COPY=tlx_projection.COPY_MODE, BN=tlx_projection.BN_S4,
                                   DEPTH=tlx_projection.DEPTH_S4)
    if fused:
        mxo = {} if mx is None else dict(zip(("YQ", "YS", "YQ2", "YS2"), mx))
        grid = 72 + 64 * (mx is not None)
        rs_args = (*ra[:-1], *sa[1:4], ws.state.status, failure)
        rs_kw = dict(**so, EARLY=EARLY_RS, **mxo)
        if CODE_WARM_RS:
            rs_kw.update(_code_operands(rows_slots, rs_args, rs_kw, grid, failure.device))
        rows_slots[(grid,)](*rs_args, **rs_kw)
    else:
        finish_rows[(8,)](*ra, **ro)
        slots[(64,)](*sa, **so)
    return ws.scores, ws.ids, ws.pooled, ws.sorted_ids, ws.expert_ids, ws.num_post

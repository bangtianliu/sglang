"""TLX S5 A16W4 expert phases, exact post-norm, weighted sum and virtual AR.

No activation quantization, store pins, physical peers or unbounded waits.
"""

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard,
    tlx_memory,
    tlx_standin,
    tlx_workspace,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW2,
    require_pin_support,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_attention import (
    fixed,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_core import (
    expert_tile,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_wide import (
    expert_tile_wide,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_bf16 import (
    expert_tile_bf16,
)

_make_common_workspace = tlx_workspace.make_s5


# Expert phases read their first task's slot table with the failure word.
EARLY_ISSUE = False
# K512 banks in flight per wide expert task (W13 / W2).
DEPTH13 = 2
DEPTH2 = 2
# probe: int64 [64] buffer for sum_reduce phase stamps (mk/trace_sum.py)
SUM_TRACE = None
# probe: int64 [64] exchange stamps of sum_reduce's rows (see tlx_attn_tail.S3_XTRACE)
S5_XTRACE = None
# probe: (w13, w2) int64 [1728 * 2] buffers for per-CTA start/end stamps (mk/trace_experts.py)
PHASE_TRACE = None
# host AR: a sibling CTA per row publishes the second half of the payload (grid 16)
HOST_SPLIT = 2
# sum_reduce (fused-norm path) issues every route's loads before the failure read
EARLY_SUM = True
# waves per EU for the wide W13 / W2 phases
WPE13 = 3
# MXFP8 activations x MXFP4 weights on the scaled WMMA (new numerics, user-approved 2026-11-19).
# Default since MK.66 (2026-11-20): MX13 + SPLIT13 + FOLD13 = W13 reads its bf16 input as two
# exact fp8 terms folded into one WMMA (same error vs FP64 as the bf16 path, -5 to -7% per layer
# on 4 GPUs). All three False = the bf16 path that matches the reference archive bit for bit.
MX13 = True
MX2 = False
# W13 activations as two MXFP8 terms (hi + residual): bf16 x4 is represented exactly
SPLIT13 = True
# MK.62 (with MX13 + SPLIT13): both fp8 terms in ONE scaled WMMA, hi in tile rows 0-7 and lo in
# rows 8-15 (a block never has more than 8 valid rows at 8 tokens)
FOLD13 = True
# MK.67 (needs FOLD13): W13 writes its output as two exact fp8 terms and W2 folds them the same way
FOLD2 = False
# W2 at 3 CTAs per CU with a 2-deep ring: the busiest rank's W2 tasks then run in one resident
# round (MK.50: -0.4..-3.7% per layer on 4 GPUs, archive-exact; tuned on GPU0 it was 4 / 2)
WPE2 = 3
# MK.41: W13 and W2 in one persistent launch; W2 tasks of a block wait for that block's
# 64 W13 tiles, published on the atomic path (exact: same tile math)
FUSE13_2 = False
# Grid of the unfused W13 / W2 launches. None = one resident round, 3 CTAs per CU (both phases
# run 3 per CU), capped at the 1728 status entries: 576 on 192 CUs, 768 on 256. Below 1728 the
# launch is persistent: CTA pid runs tasks pid, pid + grid, ...
EXPERT_GRID = None
FUSED_GRID = 576
# BF16-expert layers (0, 61): tile K step, LDS depth and waves per EU of both phases
# (h13-16 sweep, notes/h13-16-layers-0-61.md: K256 / depth 2 / 2 per EU is best at E = 2..16)
BF16_K_STEP = 256
BF16_DEPTH = 2
BF16_WPE = 2


def expert_launch_grid(device):
    """The unfused expert phases' grid: EXPERT_GRID, or 3 CTAs per CU capped at 1728."""
    if EXPERT_GRID is not None:
        return EXPERT_GRID
    return min(1728, 3 * torch.cuda.get_device_properties(device).multi_processor_count)
FUSED_WPE = 3
FUSED_DEPTH2 = 2
FUSED_PREWAIT = True
PROBE_PUBLISH13 = False  # probe: unfused W13 publishes by atomics (cost of the atomic path)
FUSED_POLL_SLEEP = 0
PUBLISH64 = False
# probe: W2 waits for ALL W13 tiles of the launch (no W13/W2 overlap in time)
FUSED_GLOBAL = False
# probe (MK.50): the fused binary in two launches (W13 tasks, then W2 tasks)
FUSED_SPLIT = False
# probes (MK.50, valid only with FUSED_SPLIT): NOSYNC = W2 runs the plain unfused W2 tile call
# (no poll/done/clears); NOPUBLISH (needs NOSYNC) = W13 stores with plain stores, no atomics
FUSED_NOSYNC = False
FUSED_NOPUBLISH = False
# probe (MK.50, needs SPLIT+NOSYNC+NOPUBLISH): the fused binary runs one phase, expert_phase the other
# (1: fused W13 then expert_phase W2; 2: expert_phase W13 then fused W2)
FUSED_MIX = 0
# probe (MK.50): N never-executed W13 tile bodies between the two loops (moves the W2 body's code)
FUSED_PAD = 0
# probe (MK.50): 1 = the fused kernel loads weight tiles non-temporally (".cs"), like gfx950's .cg on
# expert weights (keeps the stream from evicting later stages' lines); 0 = default policy
FUSED_WEIGHT_NT = 0
FUSED_TRACE = None  # probe: int64 [grid * 4] (start, W13 done, W2 wait ticks, end)
FUSED_BOUND_TICKS = 200000  # 2 ms of the 100 MHz realtime clock, then fail


@triton.jit
def expert_fused(
    X, Sorted, Experts, NumPost, W13, S13, Inter, W2, S2, Out, Cnt, Status, Failure, Phase,
    GRID: tl.constexpr, SKIP: tl.constexpr = -1, Seed=None, DEPTH13: tl.constexpr = 2,
    DEPTH2: tl.constexpr = 4, BOUND: tl.constexpr = 200000, Trace=None,
    PREWAIT: tl.constexpr = False,
    POLL_SLEEP: tl.constexpr = 0,
    PUBLISH64: tl.constexpr = False,
    GLOBAL: tl.constexpr = False,
    WCACHE: tl.constexpr = "",
    NOSYNC: tl.constexpr = False,
    NOPUBLISH: tl.constexpr = False,
    PAD: tl.constexpr = 0,
):
    """Persistent W13 then W2 (resident grid). Cnt [blocks * 32] arrivals, Cnt + 4096 the
    W2 tasks done per block; the last W2 task of a block clears both for the next launch."""
    pid = tl.program_id(0)
    allowed = tlx_guard.enabled(Failure, 5)
    ok = allowed & (pid != SKIP)
    tlx_guard.delay(Seed, pid, 0)
    TRACE: tl.constexpr = Trace is not None
    s0 = tlx_standin.clock() if TRACE else 0
    s2 = s0
    tasks = tl.load(NumPost) // 16 * 64
    if ok & (Phase != 3):  # Phase: 0 both, 2 W13 only, 3 W2 only (never 1: Triton specializes it)
        for task in range(pid, tasks, GRID):
            block = task // 64
            expert = tl.load(Experts + block)
            if NOPUBLISH:
                expert_tile_wide(X, Sorted, W13, S13, Inter, block, expert, task % 64, True,
                                 WCACHE, True, False, DEPTH=DEPTH13)
            else:
                expert_tile_wide(X, Sorted, W13, S13, Inter, block, expert, task % 64, True,
                                 WCACHE, True, False, DEPTH=DEPTH13, Publish=Cnt,
                                 PUBLISH64=PUBLISH64, CNT_INDEX=0 if GLOBAL else None)
    for _pad in tl.static_range(PAD):
        if Phase == 77 + _pad:  # never passed: code that only shifts the W2 body
            expert_tile_wide(X, Sorted, W13, S13, Inter, 0, tl.load(Experts), _pad, True, "",
                             True, False, DEPTH=DEPTH13)
    tlx_guard.delay(Seed, pid, 1)
    s1 = tlx_standin.clock() if TRACE else 0
    waited = s1 * 0
    good = ok
    if ok & (Phase != 2):
        for task in range(pid, tasks, GRID):
            block = task // 64
            expert = tl.load(Experts + block)
            if NOSYNC:
                expert_tile_wide(Inter, Sorted, W2, S2, Out, block, expert, task % 64, False,
                                 WCACHE, True, True, DEPTH=DEPTH2)
            else:
                t0 = tlx_standin.clock()
                if PREWAIT:
                    # W2 weight banks stream while this task waits for its block
                    seen = expert_tile_wide(Inter, Sorted, W2, S2, Out, block, expert, task % 64,
                                            False, WCACHE, True, True, DEPTH=DEPTH2, Wait=Cnt,
                                            WAIT_BOUND=BOUND, POLL_SLEEP=POLL_SLEEP,
                                            CNT_INDEX=0 if GLOBAL else None,
                                            NEED=tasks if GLOBAL else None)
                    c = 64 * seen.to(tl.int32)
                else:
                    c = tl.atomic_add(Cnt + block * 32, 0, sem="relaxed", scope="gpu")
                    while (c < 64) & ((tlx_standin.clock() - t0) < BOUND):
                        c = tl.atomic_add(Cnt + block * 32, 0, sem="relaxed", scope="gpu")
                if TRACE:
                    waited += tlx_standin.clock() - t0
                if c >= 64:
                    if not PREWAIT:
                        dep = tl.minimum(c - 64, 0)  # orders the A reads after the poll
                        expert_tile_wide(Inter + dep * 4096, Sorted, W2, S2, Out, block, expert,
                                         task % 64, False, WCACHE, True, True, DEPTH=DEPTH2)
                    ridx = 0 if GLOBAL else block
                    done = tl.atomic_add(Cnt + 4096 + ridx * 32, 1, sem="relaxed", scope="gpu")
                    if done == (tasks - 1 if GLOBAL else 63):
                        tl.atomic_xchg(Cnt + ridx * 32, 0, sem="relaxed", scope="gpu")
                        tl.atomic_xchg(Cnt + 4096 + ridx * 32, 0, sem="relaxed", scope="gpu")
                else:
                    good = False
    if TRACE:
        s3 = tlx_standin.clock()
        q = tl.arange(0, 4)
        tl.store(Trace + pid * 4 + q, tl.where(q == 0, s0, tl.where(q == 1, s1, tl.where(
            q == 2, waited, s3))).to(tl.int64))
    # the S5 status contract covers 1728 entries (the unfused W13/W2 grid)
    for j in tl.static_range((1728 + GRID - 1) // GRID):
        if pid + j * GRID < 1728:
            tl.store(Status + pid + j * GRID, good)
    if allowed:
        tlx_guard.finish(Failure, good, pid, 5)


@triton.jit
def expert_phase(
    X,
    Sorted,
    Experts,
    NumPost,
    W,
    Scale,
    Out,
    Status,
    Failure,
    GRID: tl.constexpr,
    FIRST: tl.constexpr,
    SKIP: tl.constexpr = -1,
    Seed=None,
    WEIGHT_CACHE: tl.constexpr = "",
    SCALE_ASYNC: tl.constexpr = False,
    CROSS_LAUNCH_A: tl.constexpr = False,
    WIDE: tl.constexpr = False,
    EARLY: tl.constexpr = False,
    DEPTH: tl.constexpr = 2,
    Trace=None,
    MX: tl.constexpr = False,
    XS=None,
    QOUT: tl.constexpr = False,
    OutS=None,
    X2=None,
    XS2=None,
    Publish=None,
    FOLD: tl.constexpr = False,
    # bf16 [N, 4096] weights with no scales (layers 0 and 61); Scale is unused
    WBF16: tl.constexpr = False,
    K_STEP: tl.constexpr = 256,
):
    pid = tl.program_id(0)
    TRACE: tl.constexpr = Trace is not None
    t0 = tlx_standin.clock() if TRACE else 0
    if EARLY:
        # the slot-table reads of this CTA's first task go out with the
        # failure-word read instead of after it (one round trip, not three)
        rows0 = tl.arange(0, 16)
        num0 = tl.load(NumPost)
        expert0 = tl.load(Experts + pid // 64)
        slots0 = tl.load(Sorted + (pid // 64) * 16 + rows0)
    allowed = tlx_guard.enabled(Failure, 5) if FIRST else tl.load(Failure) == 0
    ok = allowed & (pid != SKIP)
    tlx_guard.delay(Seed, pid, 0)
    if ok:
        if EARLY:
            tasks = num0 // 16 * 64
        else:
            tasks = tl.load(NumPost) // 16 * 64
        for task in range(pid, tasks, GRID):
            block = task // 64
            if EARLY:
                later = task != pid
                expert = tl.load(Experts + block, mask=later, other=expert0)
                pre = tl.load(Sorted + block * 16 + rows0, mask=later & (rows0 < 16), other=slots0)
            else:
                expert = tl.load(Experts + block)
                pre = None
            if WBF16:
                expert_tile_bf16(
                    X,
                    Sorted,
                    W,
                    Out,
                    block,
                    expert,
                    task % 64,
                    FIRST,
                    CROSS_LAUNCH_A,
                    DEPTH=DEPTH,
                    K_STEP=K_STEP,
                    SLOTS=pre,
                )
            elif WIDE:
                expert_tile_wide(
                    X,
                    Sorted,
                    W,
                    Scale,
                    Out,
                    block,
                    expert,
                    task % 64,
                    FIRST,
                    WEIGHT_CACHE,
                    SCALE_ASYNC,
                    CROSS_LAUNCH_A,
                    SLOTS=pre,
                    DEPTH=DEPTH,
                    MX=MX,
                    XS=XS,
                    QOUT=QOUT,
                    OutS=OutS,
                    X2=X2,
                    XS2=XS2,
                    Publish=Publish,
                    FOLD=FOLD,
                )
            else:
                expert_tile(
                    X,
                    Sorted,
                    W,
                    Scale,
                    Out,
                    block,
                    expert,
                    task % 64,
                    FIRST,
                    WEIGHT_CACHE,
                    SCALE_ASYNC,
                    CROSS_LAUNCH_A,
                )
    tlx_guard.delay(Seed, pid, 1)
    if GRID == 1728:
        tl.store(Status + pid, ok)
    else:
        # the S5 status contract covers 1728 entries
        for j in tl.static_range((1728 + GRID - 1) // GRID):
            if pid + j * GRID < 1728:
                tl.store(Status + pid + j * GRID, ok)
    if allowed:
        tlx_guard.finish(Failure, ok, pid, 5)
    if TRACE:
        t1 = tlx_standin.clock()
        k = tl.arange(0, 2)
        tl.store(Trace + pid * 2 + k, tl.where(k == 0, t0, t1).to(tl.int64))


@triton.jit
def norm_phase(
    Y, Gamma, Ids, Scaled, Inv, Status, Failure, RANK: tl.constexpr, Seed=None
):
    hit = tl.program_id(0)
    ok = tl.load(Failure) == 0
    if ok:
        expert = tl.load(Ids + hit)
        if (expert >= RANK * 64) & (expert < (RANK + 1) * 64):
            d = tl.arange(0, 4096)
            y = fixed(tl.load(Y + (RANK * 64 + hit) * 4096 + d).to(tl.float32), ROW2)
            gamma = fixed(tl.load(Gamma + d), ROW2)
            effective = (gamma.to(tl.float32) + 1.0).to(gamma.dtype).to(tl.float32)
            y = (y * effective).to(tl.bfloat16).to(tl.float32)
            inv = 1.0 / tl.sqrt(tl.sum(fixed(y * y, ROW2), 0) / 4096.0 + 1e-8)
            tl.store(Scaled + (RANK * 64 + hit) * 4096 + d, y)
            tl.store(Inv + (RANK * 64 + hit) * 32, inv)
    tlx_guard.delay(Seed, hit, 2)
    if not ok:
        tl.store(Status + hit, 0)


@triton.jit
def sum_reduce(
    Ids,
    Scores,
    Scaled,
    Inv,
    Weighted,
    Partial,
    Scratch,
    Peers,
    Reduced,
    Epochs,
    Status,
    Failure,
    Out,
    GRID: tl.constexpr,
    RANK: tl.constexpr,
    PAYLOAD: tl.constexpr,
    FLAGS: tl.constexpr,
    Seed=None,
    Y=None,
    Gamma=None,
    FUSE_NORM: tl.constexpr = False,
    Trace=None,
    EARLY: tl.constexpr = False,
    SPLIT: tl.constexpr = 0,
    XStamps=None,
    Mine=None,
):
    """SPLIT = parts per row (host AR only, grid 8 * SPLIT): CTA 8p + r recomputes
    row r (same math, the same bits stored again) and publishes part p."""
    pid = tl.program_id(0)
    row = pid % 8
    TRACE: tl.constexpr = Trace is not None
    t0 = tlx_standin.clock() if TRACE else 0
    tl.static_assert(not EARLY or FUSE_NORM)
    if EARLY:
        # every route's ids / score / expert-output row goes out before the
        # failure-word read (rows of other ranks' experts are masked, not read);
        # the per-route math and the accumulation order below are unchanged
        i8 = tl.arange(0, 8)
        ids8 = tl.load(Ids + row * 8 + i8)
        if Mine is not None:  # replicated experts: the slots kernel wrote the assignment
            mine8 = tl.load(Mine + row * 8 + i8) > 0
        else:
            mine8 = (ids8 >= RANK * 64) & (ids8 < (RANK + 1) * 64)
        d0 = tl.arange(0, 4096)
        gamma0 = tl.load(Gamma + d0)
        pre = ()
        for slot in tl.static_range(8):
            mine_ = tl.sum(tl.where(i8 == slot, mine8.to(tl.int32), 0), 0) > 0
            pre += (tl.load(Y + (RANK * 64 + row * 8 + slot) * 4096 + d0, mask=mine_,
                            other=0.0),)
    tag = tlx_standin.advance(Epochs + pid // 8 * (32 // max(SPLIT, 1)), row)
    ok = tl.load(Failure) == 0
    t1 = tlx_standin.clock() if TRACE else 0
    t2 = t1
    t3 = t1
    t4 = t1
    if ok:
        d = tl.arange(0, 4096)
        acc = fixed(tl.zeros((4096,), tl.float32), ROW2)
        for slot in tl.static_range(8):
            hit = row * 8 + slot
            if EARLY:
                mine = tl.sum(tl.where(i8 == slot, mine8.to(tl.int32), 0), 0) > 0
            else:
                expert = tl.load(Ids + hit)
                mine = (expert >= RANK * 64) & (expert < (RANK + 1) * 64)
                if Mine is not None:
                    mine = tl.load(Mine + hit) > 0
            if mine:
                if FUSE_NORM:
                    if EARLY:
                        y = fixed(pre[slot].to(tl.float32), ROW2)
                        gamma = fixed(gamma0, ROW2)
                    else:
                        y = fixed(
                            tl.load(Y + (RANK * 64 + hit) * 4096 + d).to(tl.float32), ROW2
                        )
                        gamma = fixed(tl.load(Gamma + d), ROW2)
                    effective = (
                        (gamma.to(tl.float32) + 1.0).to(gamma.dtype).to(tl.float32)
                    )
                    y = (y * effective).to(tl.bfloat16).to(tl.float32)
                    inv = 1.0 / tl.sqrt(tl.sum(fixed(y * y, ROW2), 0) / 4096.0 + 1e-8)
                    tl.store(Scaled + (RANK * 64 + hit) * 4096 + d, y)
                    tl.store(Inv + (RANK * 64 + hit) * 32, inv)
                    tlx_guard.delay(Seed, hit, 2)
                else:
                    y = fixed(
                        tl.load(
                            Scaled + (RANK * 64 + hit) * 4096 + d, cache_modifier=".cg"
                        ).to(tl.float32, bitcast=True),
                        ROW2,
                    )
                    inv = tl.load(
                        Inv + (RANK * 64 + hit) * 32, cache_modifier=".cg"
                    ).to(tl.float32, bitcast=True)
                score = tl.load(Scores + hit)
                acc += (score * inv) * y
        if TRACE:
            t2 = tlx_standin.clock()
        tl.store(Weighted + (RANK * 8 + row) * 4096 + d, acc)
        tl.store(Partial + (RANK * 8 + row) * 4096 + d, acc.to(tl.bfloat16))
        tlx_memory.drain_stores()
        if TRACE:
            t3 = tlx_standin.clock()
        tlx_guard.delay(Seed, row, 3)
        if SPLIT and pid >= 8:
            tlx_standin.publish_part(Partial, Peers, tag, row, pid // 8, SPLIT, GRID, RANK,
                                     WIDTH=4096)
        else:
            value, ok = tlx_standin.reduce_row(
                Partial,
                Scratch,
                Peers,
                Reduced,
                tag,
                row,
                GRID,
                RANK,
                PAYLOAD,
                FLAGS,
                WIDTH=4096,
                SPLIT=SPLIT,
                XStamps=XStamps,
            )
            if ok:
                tl.store(Out + row * 4096 + d, value.to(tl.bfloat16))
        if TRACE:
            t4 = tlx_standin.clock()
    if pid < 8:
        tl.store(Status + row, ok)
        tlx_guard.finish(Failure, ok, row, 5)
    if TRACE:
        t5 = tlx_standin.clock()
        i = tl.arange(0, 8)
        stamp = tl.where(i == 0, t0, tl.where(i == 1, t1, tl.where(i == 2, t2, tl.where(
            i == 3, t3, tl.where(i == 4, t4, t5)))))
        tl.store(Trace + row * 8 + i, stamp.to(tl.int64), i < 6)


class Workspace(tlx_workspace.S5Workspace):
    local_failure: object = None
    layer_out: object = None
    kernels: dict = msgspec.field(default_factory=dict)
    scale_async: bool = False
    fuse_norm_sum: bool = False
    wide_w13: bool = True
    wide_w2: bool = True
    fuse_cnt: object = None


def make_workspace(device):
    b = _make_common_workspace(device)
    b.state.status = torch.empty((1728,), device=device, dtype=torch.int32)
    b.state.grid = 1728
    b.state.epochs = torch.zeros((1728, 32), device=device, dtype=torch.int32)
    return Workspace(
        b.inter,
        b.output,
        b.scaled,
        b.inv,
        b.weighted,
        b.partial,
        b.state,
        b.k_partial,
        b.peers,
        torch.zeros((2,), device=device, dtype=torch.int32),
        torch.empty((8, 4096), device=device, dtype=torch.bfloat16),
    )


MX_BUFFERS = {}


def mx_buffers(device):
    """MXFP8 activation buffers (values as uint8, e8m0 scales per 32 along K):
    xq/xs = S4's pooled rows (W13 input), iq/is = W13 output (W2 input)."""
    key = str(device)
    if key not in MX_BUFFERS:
        u8 = dict(device=device, dtype=torch.uint8)
        # hi and lo terms share one buffer ([hi rows 0-7; lo rows 8-15]) so FOLD13 reads both
        # from one base pointer; xq / xq2 (and xs / xs2) are views of it
        xq12 = torch.zeros((16, 4096), **u8)
        xs12 = torch.zeros((16, 128), **u8)
        MX_BUFFERS[key] = dict(
            xq=xq12[:8], xs=xs12[:8], xq2=xq12[8:], xs2=xs12[8:], xq12=xq12, xs12=xs12,
            # per rank: [hi slots 0-63; lo slots 64-127] (the lo half is used by FOLD2 only)
            iq=torch.zeros((4, 128, 4096), **u8), **{"is": torch.zeros((4, 128, 128), **u8)},
        )
    return MX_BUFFERS[key]


def s5(
    x,
    ids,
    scores,
    table,
    weights,
    gamma,
    ws,
    *,
    rank,
    skip=-1,
    delay_seed=None,
    compile_only=False,
    trace=None,
    read_mode=2,
    w2_sk=1,
    task_trace=None,
    failure=None,
    guard_mode=0,
    output=None,
    weight_cache="",
):
    require_pin_support()
    mx = mx_buffers(x.device)
    assert (
        trace is None
        and task_trace is None
        and read_mode == 2
        and w2_sk == 1
        and guard_mode == 0
    )
    assert weight_cache in ("", ".cs")
    s = ws.state
    p = ws.peers
    failure = ws.local_failure if failure is None else failure
    output = ws.layer_out if output is None else output
    sorted_ids, experts, num_post = table
    expert_grid = expert_launch_grid(x.device)
    common = dict(
        GRID=expert_grid,
        num_warps=4,
        waves_per_eu=9,
        enable_fp_fusion=False,
        SCALE_ASYNC=ws.scale_async,
    )
    wbf16 = weights["w13"].dtype == torch.bfloat16
    a1 = (
        x,
        sorted_ids,
        experts,
        num_post,
        weights["w13"],
        weights["w13"] if wbf16 else weights["w13_scale"],
        ws.inter[rank],
        s.status,
        failure,
    )
    a2 = (
        ws.inter[rank],
        sorted_ids,
        experts,
        num_post,
        weights["w2"],
        weights["w2"] if wbf16 else weights["w2_scale"],
        ws.output[rank],
        s.status,
        failure,
    )
    k1 = dict(
        **common, FIRST=True, SKIP=skip, Seed=delay_seed, WEIGHT_CACHE=weight_cache,
        **({} if PHASE_TRACE is None else dict(Trace=PHASE_TRACE[0])),
    )
    if ws.wide_w13 and weight_cache == "":
        k1.update(WIDE=True, SCALE_ASYNC=True, waves_per_eu=WPE13, EARLY=EARLY_ISSUE, DEPTH=DEPTH13)
        if MX13:
            k1.update(MX=True, XS=mx["xs"])
            if FOLD13:
                assert SPLIT13, "FOLD13 needs the two-term quantization (SPLIT13)"
                k1.update(FOLD=True)
            elif SPLIT13:
                k1.update(X2=mx["xq2"], XS2=mx["xs2"])
        if FOLD2:
            assert FOLD13 and not MX2, "FOLD2 needs FOLD13 and replaces MX2"
        if MX2 or FOLD2:
            k1.update(QOUT=True, OutS=mx["is"][rank])
        if PROBE_PUBLISH13:
            if ws.fuse_cnt is None:
                ws.fuse_cnt = torch.zeros(8192, device=x.device, dtype=torch.int32)
            k1.update(Publish=ws.fuse_cnt)
    k2 = dict(
        **common,
        FIRST=False,
        Seed=delay_seed,
        WEIGHT_CACHE=weight_cache,
        CROSS_LAUNCH_A=True,
        **({} if PHASE_TRACE is None else dict(Trace=PHASE_TRACE[1])),
    )
    if ws.wide_w2 and weight_cache == "":
        k2.update(WIDE=True, SCALE_ASYNC=True, waves_per_eu=WPE2, EARLY=EARLY_ISSUE, DEPTH=DEPTH2)
        if MX2 or FOLD2:
            k2.update(MX=True, XS=mx["is"][rank], FOLD=bool(FOLD2))
    if wbf16:
        # bf16 pooled rows against bf16 weights: none of the MXFP8 activation paths apply
        assert not (FUSE13_2 or FOLD2 or MX2) and weight_cache == ""
        bf = {**common, "WBF16": True, "K_STEP": BF16_K_STEP, "DEPTH": BF16_DEPTH,
              "waves_per_eu": BF16_WPE, "Seed": delay_seed}
        k1 = dict(bf, FIRST=True, SKIP=skip)
        k2 = dict(bf, FIRST=False, CROSS_LAUNCH_A=True)
    else:
        if MX13:
            a1 = (mx["xq"],) + a1[1:]
        if MX2 or FOLD2:
            a1 = a1[:6] + (mx["iq"][rank],) + a1[7:]
            a2 = (mx["iq"][rank],) + a2[1:]
    an = (ws.output, gamma, ids, ws.scaled, ws.inv, s.status, failure)
    kn = dict(RANK=rank, Seed=delay_seed, num_warps=4, enable_fp_fusion=False)
    ar = (
        ids,
        scores,
        ws.scaled,
        ws.inv,
        ws.weighted,
        ws.partial,
        p.scratch,
        p.peers,
        p.reduced,
        s.epochs,
        s.status,
        failure,
        output,
    )
    kr = dict(
        GRID=p.ctas,
        RANK=rank,
        PAYLOAD=p.payload,
        FLAGS=p.flags,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
    )
    key = (
        FOLD13,
        FOLD2,
        MX2,
        expert_grid,
        PROBE_PUBLISH13,
        rank,
        skip,
        delay_seed is not None,
        gamma.dtype,
        weight_cache,
        ws.scale_async,
        ws.fuse_norm_sum,
        ws.wide_w13,
        ws.wide_w2,
    )
    if wbf16:
        key = key + ("wbf16", BF16_K_STEP, BF16_DEPTH, BF16_WPE)
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_route

    if tlx_route.REPL:
        assert ws.fuse_norm_sum and not FUSE13_2
        kr.update(Mine=tlx_route.repl_mine(sorted_ids))
        key = key + ("repl",)
    if S5_XTRACE is not None:
        kr.update(XStamps=S5_XTRACE)
    kr.update(Y=ws.output, Gamma=gamma, FUSE_NORM=ws.fuse_norm_sum, Trace=SUM_TRACE,
              EARLY=EARLY_SUM and ws.fuse_norm_sum)
    if int(HOST_SPLIT) > 1 and tlx_standin.HOST.value:
        kr.update(SPLIT=int(HOST_SPLIT))
    if FUSE13_2:
        assert ws.wide_w13 and ws.wide_w2 and weight_cache == "" and not (MX13 or MX2)
        if ws.fuse_cnt is None:
            ws.fuse_cnt = torch.zeros(8192, device=x.device, dtype=torch.int32)
        af = (x, sorted_ids, experts, num_post, weights["w13"], weights["w13_scale"],
              ws.inter[rank], weights["w2"], weights["w2_scale"], ws.output[rank], ws.fuse_cnt,
              s.status, failure)
        kf = dict(GRID=FUSED_GRID, SKIP=skip, Seed=delay_seed, DEPTH13=DEPTH13, DEPTH2=FUSED_DEPTH2,
                  BOUND=FUSED_BOUND_TICKS, Trace=FUSED_TRACE, PREWAIT=FUSED_PREWAIT, POLL_SLEEP=FUSED_POLL_SLEEP, PUBLISH64=PUBLISH64, GLOBAL=FUSED_GLOBAL, WCACHE=".cs" if FUSED_WEIGHT_NT else "", num_warps=4, waves_per_eu=FUSED_WPE,
                  enable_fp_fusion=False)
        key = key + ("fused", FUSED_GRID, FUSED_WPE, FUSED_DEPTH2, FUSED_TRACE is not None, FUSED_PREWAIT, FUSED_POLL_SLEEP, PUBLISH64, FUSED_GLOBAL, FUSED_SPLIT, FUSED_WEIGHT_NT, FUSED_NOSYNC, FUSED_NOPUBLISH, FUSED_MIX, FUSED_PAD)
        assert FUSED_SPLIT or not (FUSED_NOSYNC or FUSED_NOPUBLISH)
        assert FUSED_NOSYNC or not FUSED_NOPUBLISH
        kf.update(NOSYNC=FUSED_NOSYNC, NOPUBLISH=FUSED_NOPUBLISH, PAD=FUSED_PAD)
        if FUSED_MIX:
            assert FUSED_SPLIT and FUSED_NOSYNC and FUSED_NOPUBLISH
        if FUSED_MIX == 1:
            fused = [("w13w2", expert_fused, af + (2,), kf, FUSED_GRID),
                     ("w2", expert_phase, a2, k2, expert_grid)]
        elif FUSED_MIX == 2:
            fused = [("w13", expert_phase, a1, k1, expert_grid),
                     ("w13w2", expert_fused, af + (3,), kf, FUSED_GRID)]
        elif FUSED_SPLIT:
            fused = [("w13w2", expert_fused, af + (2,), kf, FUSED_GRID),
                     ("w13w2", expert_fused, af + (3,), kf, FUSED_GRID)]
        else:
            fused = [("w13w2", expert_fused, af + (0,), kf, FUSED_GRID)]
    phases = [
        *(fused if FUSE13_2 else [("w13", expert_phase, a1, k1, expert_grid),
                                  ("w2", expert_phase, a2, k2, expert_grid)]),
        *([] if ws.fuse_norm_sum else [("norm", norm_phase, an, kn, 64)]),
        ("sum_AR", sum_reduce, ar, kr, 8 * max(kr.get("SPLIT", 0), 1)),
    ]
    if key not in s.admitted:
        records = []
        kernels = []
        for name, fn, aa, kk, grid in phases:
            k = fn.warmup(*aa, **kk, grid=(grid,))
            r = kernel_occupancy("independent-S5-" + name, k)
            assert r.active_ctas_per_cu >= 1, (
                name,
                r,
            )
            if name == "w13w2":
                # every CTA must be resident (W2 tasks wait on other CTAs' W13 tiles)
                cus = torch.cuda.get_device_properties(x.device).multi_processor_count
                assert r.active_ctas_per_cu * cus >= grid, (name, r, cus, grid)
            records.append(
                dict(
                    phase=name,
                    vgprs=k.n_regs,
                    spills=k.n_spills,
                    lds=k.metadata.shared,
                    occupancy=r.active_ctas_per_cu,
                    waves=k.metadata.num_warps,
                    grid=grid,
                )
            )
            kernels.append(k)
        s.admitted[key] = records
        ws.kernels[key] = kernels
    if not compile_only:
        for name, fn, aa, kk, grid in phases:
            fn[(grid,)](*aa, **kk)
    return ws.kernels[key][1]

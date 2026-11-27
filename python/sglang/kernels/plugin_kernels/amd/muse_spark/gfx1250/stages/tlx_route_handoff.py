"""T9.1d small routing-row -> slot-table fence-free handoff, unselected.

Eight producers never wait. Each of64 consumers owns8 separate128B flag lines,
loads the256B ID payload once at DEV scope and stages it in LDS. Weights retain
ordinary cached loads; no release/acquire atomic or cache-wide fence is used.
"""

import msgspec
import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_memory as mem,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_attention import (
    fixed,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_route import TOP

SHARED_IDS = tl.constexpr(
    tlx.shared_linear_layout_encoding([[1], [2], [4], [8], [16], [32]], alignment=16)
)


@triton.jit
def produce(
    P,
    Bias,
    Norm,
    Scores,
    Ids,
    Y,
    Producer,
    Rows,
    Failure,
    Epoch,
    Flags,
    Seed=None,
    SKIP_ROW: tl.constexpr = -1,
):
    row = tl.program_id(0)
    check = tl.arange(0, 512)
    ok = tl.min(tl.load(Producer + check, check < 288, 1), 0) == 1
    ok = ok & (row != SKIP_ROW)
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
        tl.store(Ids + row * 8 + slots, ids, cache_modifier=".cg")
        tl.store(Scores + row * 8 + slots, selected / (total + 1e-15))
        # Payload is agent-visible and drained before private consumer flags.
        mem.drain_stores()
        consumers = tl.arange(0, 64)
        tag = tl.load(Epoch)
        mem.store_words(
            Flags + (row * 64 + consumers) * 32,
            tl.full((64,), tag, tl.int32),
            tl.full((64,), True, tl.int1),
        )
        mem.drain_stores()
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
def consume(
    Ids,
    Sorted,
    Experts,
    NumPost,
    Rows,
    Status,
    Failure,
    Epoch,
    Flags,
    Record,
    Polls,
    LOCAL_START: tl.constexpr,
    GRID: tl.constexpr,
    Seed=None,
):
    pid = tl.program_id(0)
    tag = tl.load(Epoch)
    lines = tlx.require_layout(tl.arange(0, 128), ROW1)
    mask = lines < 8
    address = Flags + (lines * 64 + pid) * 32
    seen = mem.load_words_masked(address, mask)
    ok = tl.sum(((seen == tag) & mask).to(tl.int32), 0) == 8
    retries = 0
    while (~ok) & (retries < 4096):
        tl.inline_asm_elementwise(
            "s_sleep 1\nv_mov_b32 $0,0", "=v", [], dtype=tl.int32, is_pure=False, pack=1
        )
        seen = mem.load_words_masked(address, mask)
        ok = tl.sum(((seen == tag) & mask).to(tl.int32), 0) == 8
        retries += 1
    tl.store(Polls + pid, retries)
    if not ok:
        first = tl.atomic_cas(Record, 0, 1, sem="relaxed", scope="gpu")
        if first == 0:
            missing = tl.min(tl.where(mask & (seen != tag), lines, 8), 0)
            observed = tl.sum(tl.where(lines == missing, seen, 0), 0)
            tl.store(Record + 1, pid)
            tl.store(Record + 2, missing)
            tl.store(Record + 3, tag)
            tl.store(Record + 4, observed)
            tl.store(Record + 5, retries)
    tlx_guard.delay(Seed, pid, 2)
    if ok:
        j = tl.arange(0, 64)
        staged = tlx.local_alloc((64,), tl.int32, 1, layout=SHARED_IDS)
        payload = tl.load(Ids + j, cache_modifier=".cg")
        tlx.local_store(tlx.local_view(staged, 0), payload)
        tlx.workgroup_barrier()
        raw = (
            tlx.release_layout(tlx.local_load(tlx.local_view(staged, 0), layout=ROW1))
            - LOCAL_START
        )
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
        hit = tl.min(
            tl.where(match[None, :] & (rank[None, :] == nth[:, None]), j[None, :], 64),
            1,
        )
        tl.store(Sorted + pid * 16 + d, hit)
        tl.store(Experts + pid, tl.where(owner < 64, owner, -1))
        if pid == 0:
            tl.store(NumPost, total)
    status = pid * 16 + tl.arange(0, 16)
    tl.store(Status + status, ok, status < GRID)
    tlx_guard.finish(Failure, ok, pid, 4)


class Workspace(msgspec.Struct):
    epoch: object
    flags: object
    record: object
    polls: object
    stream: object


def make_workspace(epoch):
    device = epoch.device
    return Workspace(
        epoch,
        torch.zeros(8 * 64 * 32, device=device, dtype=torch.int32),
        torch.zeros(8, device=device, dtype=torch.int32),
        torch.zeros(64, device=device, dtype=torch.int32),
        torch.cuda.Stream(device=device),
    )

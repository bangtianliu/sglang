"""TLX S2: full-chunk async LDS ring with the reference split/merge math.

All page/token-map paths use masked 64x64 FP8 async copies into swizzled LDS. Pins anchor compute and
reduction inputs only; output stores/indices are compiler-managed. GPU-local
stand-ins only; dispatch boundaries replace in-launch producer polling.
"""

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard,
    tlx_standin,
    tlx_workspace,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_memory
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_qkvg import norm_rope
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    BlockedEncoding,
    require_pin_support,
)


class AttentionMma(tlx.layout_encoding):
    def __init__(self, operand=-1):
        self.operand = operand

    def __repr__(self):
        return f"AttentionMma4_operand{self.operand}"

    def __hash__(self):
        return hash((AttentionMma, self.operand))

    def __eq__(self, other):
        return isinstance(other, AttentionMma) and self.operand == other.operand

    def to_ir(self, builder, *_):
        parent = builder.get_amd_wmma_layout(
            3, True, [[1, 0], [2, 0]], [], [], [16, 16, 32], 2
        )
        return (
            parent
            if self.operand < 0
            else builder.get_dot_operand_layout(self.operand, parent, 8)
        )


class Slice(tlx.layout_encoding):
    def __init__(self, dim, parent):
        self.dim = dim
        self.parent = parent

    def __repr__(self):
        return f"Slice{self.dim}_{self.parent}"

    def __hash__(self):
        return hash((Slice, self.dim, self.parent))

    def __eq__(self, other):
        return (
            isinstance(other, Slice)
            and self.dim == other.dim
            and self.parent == other.parent
        )

    def to_ir(self, builder, *_):
        return builder.get_slice_layout(self.dim, self.parent.to_ir(builder))


MMA = tl.constexpr(AttentionMma())
DOT_A = tl.constexpr(AttentionMma(0))
DOT_B = tl.constexpr(AttentionMma(1))
MERGE_PARENT = BlockedEncoding([1, 4], [2, 16], [4, 1], [1, 0])
MERGE = tl.constexpr(MERGE_PARENT)
MERGE_SPLIT = tl.constexpr(Slice(1, MERGE_PARENT))
# Latency-bound kernels issue their data loads with the failure-word read.
EARLY_ISSUE = False
# MK.109 (h13-01 session's patch): Length and the split's pages are loaded before the failure-word
# round trip instead of after it; the same values, so bit-exact. Off until it wins on 4 GPUs.
EARLY_PLAN = bool(int(__import__("os").environ.get("MUSE_ATTN_EARLY_PLAN", "0")))
QHEAD = tl.constexpr(BlockedEncoding([1, 2], [1, 32], [4, 1], [1, 0]))
# S3 head norm/gate carrier (one (row, head) 64-vector per CTA): shared by
# tlx_attn_tail.normalize_rows and the fused merge, so both compute one tree.
WIDE_NORM = tl.constexpr(BlockedEncoding([8], [32], [4], [0]))


@triton.jit
def head_norm_gate(x, g):
    """rmsnorm(x) * sigmoid(g) of one 64-wide head; x, g FP32 of BF16 values."""
    x = tlx.require_layout(x, WIDE_NORM)
    x = tlx.require_layout(x, WIDE_NORM, pin=True)
    g = tlx.require_layout(g, WIDE_NORM)
    g = tlx.require_layout(g, WIDE_NORM, pin=True)
    inv = tl.rsqrt(tl.sum(x * x, 0) / 64 + 1e-5)
    return (x * inv) / (1.0 + tl.exp(-g))


@triton.jit
def fixed(value, layout: tl.constexpr):
    # Type the input before anchoring, and preserve the carrier at JIT edges.
    # A pin on plain data otherwise forces default-layout input conversions.
    value = tlx.require_layout(value, layout)
    return tlx.require_layout(value, layout, pin=True)


@triton.jit
def cached_pages(
    Pages, Map, start, end, MAX_PAGES: tl.constexpr, MAP_PAGE: tl.constexpr
):
    i = tl.arange(0, MAX_PAGES)
    logical = tl.minimum(start // 16 + i, tl.maximum(tl.cdiv(end, 16) - 1, 0))
    pages = tl.load(Pages + logical)
    if MAP_PAGE:
        pages = tl.load(Map + pages * 16)
    return pages


@triton.jit
def slots_at(
    pages,
    Map,
    offset,
    start,
    end,
    MAX_PAGES: tl.constexpr,
    HAS_MAP: tl.constexpr,
    MAP_PAGE: tl.constexpr,
):
    token = offset + tl.arange(0, 64)
    page = tl.gather(
        pages, tl.minimum(tl.maximum((token - start) // 16, 0), MAX_PAGES - 1), 0
    )
    slot = page + token % 16 if MAP_PAGE else page * 16 + token % 16
    if HAS_MAP and not MAP_PAGE:
        slot = tl.load(Map + slot)
    return slot, (token < end) & (slot >= 0)


CHUNK_PARENT = BlockedEncoding([1, 16], [8, 4], [4, 1], [1, 0])
CHUNK_LOAD = tl.constexpr(CHUNK_PARENT)
CHUNK_ROW = tl.constexpr(Slice(1, CHUNK_PARENT))


@triton.jit
def copy_async_chunk(
    K,
    V,
    pages,
    Map,
    offset,
    start,
    end,
    head,
    MAX_PAGES: tl.constexpr,
    HAS_MAP: tl.constexpr,
    MAP_PAGE: tl.constexpr,
    PIN_VALID: tl.constexpr,
    sk,
    sv,
    bank,
):
    slot, valid = slots_at(pages, Map, offset, start, end, MAX_PAGES, HAS_MAP, MAP_PAGE)
    # Retain the compute carrier where cold chain A/B wins. Full8k and
    # the unmeasured full32k keep the original unanchored validity path.
    if PIN_VALID:
        valid = fixed(valid, CHUNK_ROW)
    d = tl.arange(0, 64)
    pointer = tl.maximum(slot, 0)[:, None] * 256 + head * 64 + d[None, :]
    kp = tlx.require_layout(K + pointer, CHUNK_LOAD)
    vp = tlx.require_layout(V + pointer, CHUNK_LOAD)
    kp = tl.max_contiguous(tl.multiple_of(kp, (1, 16)), (1, 16))
    vp = tl.max_contiguous(tl.multiple_of(vp, (1, 16)), (1, 16))
    mask = tlx.require_layout(tl.broadcast_to(valid[:, None], (64, 64)), CHUNK_LOAD)
    zero = tlx.require_layout(
        tl.full((64, 64), 0, tl.uint8).to(K.dtype.element_ty, bitcast=True), CHUNK_LOAD
    )
    tlx.async_load(kp, tlx.local_view(sk, bank), mask, zero)
    tlx.async_load(vp, tlx.local_view(sv, bank), mask, zero)
    tlx.async_load_commit_group()
    return tlx.release_layout(valid)


@triton.jit
def split_plan(Length, Pages, Map, split, LEFT: tl.constexpr, SPLITS: tl.constexpr,
               MAX_PAGES: tl.constexpr, MAP_PAGE: tl.constexpr):
    length = tl.load(Length)
    first = tl.maximum(0, length - 8 - LEFT) // 64 * 64 if LEFT >= 0 else 0
    tiles = tl.cdiv(length - first, 64)
    start = first + split * tiles // SPLITS * 64
    end = tl.minimum(length, first + (split + 1) * tiles // SPLITS * 64)
    pages = cached_pages(Pages, Map, start, end, MAX_PAGES, MAP_PAGE)
    return length, start, end, pages


@triton.jit
def partial(
    Q,
    K,
    V,
    Pages,
    Map,
    Length,
    KS,
    VS,
    head,
    split,
    SCALE: tl.constexpr,
    LEFT: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_MAP: tl.constexpr,
    MAP_PAGE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    DIRECT_KV: tl.constexpr = False,
    FUSE: tl.constexpr = False,
    P1=None,
    POS=None,
    COS=None,
    EPS: tl.constexpr = 1e-5,
    QSCALE: tl.constexpr = 1.0,
    ROPE: tl.constexpr = True,
    STORE_Q: tl.constexpr = False,
    LOC=None,
    KOut=None,
    HEADS_MAP: tl.constexpr = False,
    PLAN=None,
):
    r = tl.arange(0, 64)
    d = tl.arange(0, 64)
    if not FUSE:
        q = tl.load(
            Q + (r[:, None] // 8) * 2048 + (head * 8 + r[:, None] % 8) * 64 + d[None, :]
        )
        q = fixed(q, DOT_A)
    if PLAN is None:
        length, start, end, pages = split_plan(Length, Pages, Map, split, LEFT, SPLITS, MAX_PAGES, MAP_PAGE)
    else:
        length, start, end, pages = PLAN
    acc = fixed(tl.zeros((64, 64), tl.float32), MMA)
    maximum = tl.full((64,), float("-inf"), tl.float32)
    denom = tl.zeros((64,), tl.float32)
    scale = SCALE * 1.4426950408889634 * tl.load(KS)
    # A full 64-token async copy replaces four page TDM copies (and the
    # register-held token-map fallback). Physical LDS image matches Gluon.
    sh: tl.constexpr = tlx.shared_linear_layout_encoding(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [1, 16],
            [2, 32],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
        ],
        alignment=16,
    )
    sk = tlx.local_alloc((64, 64), K.dtype.element_ty, 2, layout=sh)
    sv = tlx.local_alloc((64, 64), V.dtype.element_ty, 2, layout=sh)
    score = fixed(tl.full((64, 64), float("-inf"), tl.float32), MMA)
    current_valid = tl.full((64,), False, tl.int1)
    if FUSE:
        # this launch's keys must be in the cache before any chunk is copied
        current_tokens(P1, POS, COS, LOC, Map, K, KS, KOut, length, start, end,
                       head, EPS, ROPE, HEADS_MAP)
    if start < end:
        current_valid = copy_async_chunk(
            K,
            V,
            pages,
            Map,
            start,
            start,
            end,
            head,
            MAX_PAGES,
            HAS_MAP,
            MAP_PAGE,
            LEFT >= 0 or MAX_PAGES >= 64,
            sk,
            sv,
            0,
        )
    if FUSE:
        # S1's q head norm + RoPE for this CTA's 8 query heads, computed here
        # (same tree/rounding as tlx_qkvg.qkvg_heads) while the first chunk lands.
        rq = tl.arange(64, 128) - 64  # distinct ops: CSE must not merge with r / d
        dq = tl.arange(128, 192) - 128
        qv = tl.load(
            P1 + (rq[:, None] // 8) * 4608 + (head * 8 + rq[:, None] % 8) * 64 + dq[None, :]
        ).to(tl.float32, bitcast=True)
        qv = tlx.require_layout(qv, QHEAD, pin=True)
        qn = norm_rope(qv, POS, COS, EPS, QSCALE, ROPE, True, QHEAD, 8, 256)
        if STORE_Q:
            rs = tl.arange(512, 576) - 512
            ds = tl.arange(576, 640) - 576
            tl.store(
                Q + (rs[:, None] // 8) * 2048 + (head * 8 + rs[:, None] % 8) * 64 + ds[None, :],
                qn,
            )
        q = fixed(qn, DOT_A)
    if start < end:
        tlx.async_load_wait_group(0)
        tlx.workgroup_barrier()
        if DIRECT_KV:
            kk = fixed(
                tlx.local_load(tlx.local_trans(tlx.local_view(sk, 0)), layout=DOT_B),
                DOT_B,
            ).to(tl.bfloat16)
        else:
            kk = tlx.local_load(tlx.local_view(sk, 0), layout=CHUNK_LOAD).to(
                tl.bfloat16
            )
        score = (
            tl.dot(
                q,
                fixed(kk if DIRECT_KV else tl.trans(kk), DOT_B),
                fixed(tl.zeros((64, 64), tl.float32), MMA),
            )
            * scale
        )
    iteration = 0
    for offset in range(start, end, 64):
        bank = iteration % 2
        next_valid = current_valid
        if offset + 64 < end:
            next_valid = copy_async_chunk(
                K,
                V,
                pages,
                Map,
                offset + 64,
                start,
                end,
                head,
                MAX_PAGES,
                HAS_MAP,
                MAP_PAGE,
                LEFT >= 0 or MAX_PAGES >= 64,
                sk,
                sv,
                1 - bank,
            )
        absolute = length - 8 + r // 8
        mask = (offset + d[None, :] <= absolute[:, None]) & current_valid[None, :]
        if LEFT >= 0:
            mask &= offset + d[None, :] >= absolute[:, None] - LEFT
        score = fixed(
            tl.where(tl.broadcast_to(mask, (64, 64)), score, float("-inf")), MMA
        )
        new_max = tl.maximum(maximum, tl.max(score, 1))
        safe = tl.where(new_max == float("-inf"), 0.0, new_max)
        prob = fixed(tl.exp2(score - safe[:, None]), MMA)
        alpha = tl.exp2(maximum - safe)
        denom = denom * alpha + tl.sum(prob, 1)
        acc = acc * alpha[:, None]
        next_score = score
        if offset + 64 < end:
            tlx.async_load_wait_group(0)
            tlx.workgroup_barrier()
            if DIRECT_KV:
                kk = fixed(
                    tlx.local_load(
                        tlx.local_trans(tlx.local_view(sk, 1 - bank)), layout=DOT_B
                    ),
                    DOT_B,
                ).to(tl.bfloat16)
            else:
                kk = tlx.local_load(tlx.local_view(sk, 1 - bank), layout=CHUNK_LOAD).to(
                    tl.bfloat16
                )
            next_score = (
                tl.dot(
                    q,
                    fixed(kk if DIRECT_KV else tl.trans(kk), DOT_B),
                    fixed(tl.zeros((64, 64), tl.float32), MMA),
                )
                * scale
            )
        if DIRECT_KV:
            vv = fixed(
                tlx.local_load(tlx.local_view(sv, bank), layout=DOT_B), DOT_B
            ).to(tl.bfloat16)
        else:
            vv = tlx.local_load(tlx.local_view(sv, bank), layout=CHUNK_LOAD).to(
                tl.bfloat16
            )
        acc = tl.dot(
            fixed(prob.to(tl.bfloat16), DOT_A), fixed(vv, DOT_B), fixed(acc, MMA)
        )
        maximum = new_max
        score = next_score
        current_valid = next_valid
        tlx.workgroup_barrier()
        iteration += 1
    value = acc * (tl.load(VS) / tl.where(denom > 0, denom, 1.0))[:, None]
    lse = tl.where(denom > 0, maximum + tl.log2(denom), float("-inf"))
    return value, lse


@triton.jit
def partials(
    Q,
    K,
    V,
    Pages,
    Map,
    Length,
    KS,
    VS,
    P,
    LSE,
    Epochs,
    Status,
    Failure,
    SPLITS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    SCALE: tl.constexpr,
    LEFT: tl.constexpr,
    HAS_MAP: tl.constexpr,
    MAP_PAGE: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    SKIP: tl.constexpr = -1,
    PIPELINED: tl.constexpr = False,
    Seed=None,
    DIRECT_KV: tl.constexpr = False,
    FUSE_HEADS: tl.constexpr = False,
    P1=None,
    POS=None,
    COS=None,
    LOC=None,
    KOut=None,
    EPS: tl.constexpr = 1e-5,
    QSCALE: tl.constexpr = 1.0,
    ROPE: tl.constexpr = True,
    HEADS_MAP: tl.constexpr = False,
    EARLY_PLAN: tl.constexpr = False,
):
    pid = tl.program_id(0)
    plan = None
    if EARLY_PLAN:
        # Length and this split's pages are in-bounds for every pid: load them before the
        # failure-word round trip instead of after it (bit-exact, same values).
        plan = split_plan(Length, Pages, Map, pid // 4, LEFT, SPLITS, MAX_PAGES, MAP_PAGE)
    tag = tlx_standin.advance(Epochs, pid)
    allowed = tlx_guard.enabled(Failure, 2)
    ok = allowed & (pid != SKIP)
    tlx_guard.delay(Seed, pid, 0)
    if ok:
        split = pid // 4
        head = pid % 4
        value, lse = partial(
            Q,
            K,
            V,
            Pages,
            Map,
            Length,
            KS,
            VS,
            head,
            split,
            SCALE,
            LEFT,
            SPLITS,
            HAS_MAP,
            MAP_PAGE,
            MAX_PAGES,
            POOL_TOKENS,
            DIRECT_KV,
            FUSE_HEADS,
            P1,
            POS,
            COS,
            EPS,
            QSCALE,
            ROPE,
            split == 0,
            LOC,
            KOut,
            HEADS_MAP,
            plan,
        )
        row = tl.arange(0, 64)
        col = tl.arange(0, 64)
        tl.store(
            P + ((head * 64 + row[:, None]) * SPLITS + split) * 64 + col[None, :],
            value.to(tl.int32, bitcast=True),
        )
        tl.store(
            LSE + (head * 64 + row) * SPLITS + split, lse.to(tl.int32, bitcast=True)
        )
    tlx_guard.delay(Seed, pid, 1)
    tl.store(Status + pid, ok)
    if allowed:
        tlx_guard.finish(Failure, ok, pid, 2)


@triton.jit
def current_tokens(P1, POS, COS, LOC, Map, KC, KS, KOut, length, start, end, head,
                   EPS: tl.constexpr, ROPE: tl.constexpr, HAS_MAP: tl.constexpr):
    """S1's k norm + RoPE and k-cache write for the launch's tokens that fall in
    this split. Every CTA of the split writes all four KV heads of its tokens
    (identical bytes), then drains, so no CTA reads a cache line it has not
    itself completely written first. KOut gets this CTA's own head."""
    if (end > length - 8) & (start < end):
        r = tl.arange(1024, 1056) - 1024  # row = token * 4 + kv head
        d = tl.arange(1056, 1120) - 1056
        kv = tl.load(
            P1 + (r[:, None] // 4) * 4608 + 2048 + (r[:, None] % 4) * 64 + d[None, :]
        ).to(tl.float32, bitcast=True)
        kv = tlx.require_layout(kv, QHEAD, pin=True)
        kn = norm_rope(kv, POS, COS, EPS, 1.0, ROPE, False, QHEAD, 4, 2048)
        tok = r // 4
        position = length - 8 + tok
        mine = (position >= start) & (position < end)
        slot = tl.load(LOC + tok, mine, -1).to(tl.int64)
        if HAS_MAP:
            slot = tl.load(Map + slot, mine & (slot >= 0), -1).to(tl.int64)
        tl.store(
            KC + slot[:, None] * 256 + (r[:, None] % 4) * 64 + d[None, :],
            kn.to(tl.float32) / tl.load(KS),
            (mine & (slot >= 0))[:, None],
        )
        tl.store(
            KOut + tok[:, None] * 256 + (r[:, None] % 4) * 64 + d[None, :],
            kn,
            (mine & (r % 4 == head))[:, None],
        )
        tlx_memory.drain_stores()


@triton.constexpr_function
def split_extent(value):
    return 1 << (value - 1).bit_length()


@triton.jit
def merge(
    P,
    LSE,
    Status,
    CtaStatus,
    Out,
    Failure,
    SPLITS: tl.constexpr,
    Seed=None,
    CACHE: tl.constexpr = ".cg",
    Gate=None,
    Norm=None,
    FUSE_NORM: tl.constexpr = False,
    EARLY: tl.constexpr = False,
):
    pid = tl.program_id(0)
    SL: tl.constexpr = split_extent(SPLITS)
    ss = tl.arange(0, SL)
    dd = tl.arange(0, 64)
    if EARLY:
        # the partials go out with the failure-word read, not after it
        val0 = tl.load(
            P + (pid * SPLITS + tl.minimum(ss[:, None], SPLITS - 1)) * 64 + dd[None, :],
            cache_modifier=CACHE,
        )
        lse0 = tl.load(
            LSE + pid * SPLITS + tl.minimum(ss, SPLITS - 1),
            ss < SPLITS,
            0,
            cache_modifier=CACHE,
        )
        if FUSE_NORM:
            dg = tlx.require_layout(tl.arange(64, 128) - 64, WIDE_NORM)
            gate0 = tl.load(
                Gate + (pid % 64 // 8) * 2048 + (pid // 64 * 8 + pid % 8) * 64 + dg
            )
    ok = tl.load(Failure) == 0
    tlx_guard.delay(Seed, pid, 2)
    if ok:
        if EARLY:
            val = val0.to(tl.float32, bitcast=True)
        else:
            val = tl.load(
                P + (pid * SPLITS + tl.minimum(ss[:, None], SPLITS - 1)) * 64 + dd[None, :],
                cache_modifier=CACHE,
            ).to(tl.float32, bitcast=True)
        val = fixed(val, MERGE)
        if EARLY:
            lse = lse0.to(tl.float32, bitcast=True)
        else:
            lse = tl.load(
                LSE + pid * SPLITS + tl.minimum(ss, SPLITS - 1),
                ss < SPLITS,
                0,
                cache_modifier=CACHE,
            ).to(tl.float32, bitcast=True)
        lse = fixed(tl.where(ss < SPLITS, lse, float("-inf")), MERGE_SPLIT)
        maximum = tl.max(lse, 0)
        maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
        weight = tl.exp2(lse - maximum)
        total = tl.sum(weight, 0)
        weighted = fixed(val * weight[:, None], MERGE)
        result = tl.sum(weighted, 0) / tl.where(total > 0, total, 1.0)
        tl.store(
            Out + (pid % 64 // 8) * 2048 + (pid // 64 * 8 + pid % 8) * 64 + dd,
            result.to(tl.bfloat16),
        )
        if FUSE_NORM:
            # S3's head norm/gate on this CTA's own (row, head): same BF16 input
            # bits as the stored attention output, same tree as normalize_rows.
            dn = tlx.require_layout(dd, WIDE_NORM)
            index = (pid % 64 // 8) * 2048 + (pid // 64 * 8 + pid % 8) * 64 + dn
            x = result.to(tl.bfloat16).to(tl.float32)
            if EARLY:
                g = gate0.to(tl.float32)
            else:
                g = tl.load(Gate + index).to(tl.float32)
            tl.store(Norm + index, head_norm_gate(x, g).to(tl.bfloat16))
    else:
        if pid < SPLITS * 4:
            tl.store(CtaStatus + pid, 0)
        if pid + 256 < SPLITS * 4:
            tl.store(CtaStatus + pid + 256, 0)
    tl.store(Status + pid, ok)


class Workspace(tlx_workspace.S2Workspace):
    local_failure: object = None


def make_workspace(device, *, context, window_left=-1, **kwargs):
    b = tlx_workspace.make_s2(device, context=context, window_left=window_left)
    b.ctas = b.splits * 4
    return Workspace(
        b.partial,
        b.lse,
        torch.zeros((b.ctas, 32), device=device, dtype=torch.int32),
        b.counters,
        b.flags,
        b.status,
        torch.empty((b.ctas,), device=device, dtype=torch.int32),
        b.splits,
        b.ctas,
        b.pages_per_chunk,
        b.shape,
        {},
        torch.zeros((2,), device=device, dtype=torch.int32),
    )


def attention(
    q,
    k,
    v,
    page_table,
    length,
    k_descale,
    v_descale,
    *,
    context,
    window_left=-1,
    mapping=None,
    mapping_page_aligned=False,
    workspace=None,
    ctas=None,
    block_n=64,
    waves=4,
    softmax_scale=0.125,
    skip=-1,
    delay_seed=None,
    shape=None,
    output=None,
    failure=None,
    guard_mode=0,
    direct_kv=None,
    merge_cache=None,
    norm=None,
    heads=None,
):
    assert guard_mode == 0 and block_n == 64 and waves == 4
    require_pin_support()
    ws = (
        make_workspace(q.device, context=context, window_left=window_left)
        if workspace is None
        else workspace
    )
    failure = ws.local_failure if failure is None else failure
    output = torch.empty_like(q) if output is None else output
    if direct_kv is None:
        # Keep unmeasured contexts on their original kernel configuration.
        direct_kv = (context, window_left) in (
            (8192, 2047),
            (131072, -1),
            (131072, 2047),
        )
    if merge_cache is None:
        # Partials are published by same-stream kernel completion. Default-cache
        # reads win on these sliding shapes; full attention keeps its old policy.
        merge_cache = (
            "" if (context, window_left) in ((8192, 2047), (131072, 2047)) else ".cg"
        )
    assert merge_cache in ("", ".cg")
    pa = (
        q,
        k,
        v,
        page_table,
        mapping,
        length,
        k_descale,
        v_descale,
        ws.partial,
        ws.lse,
        ws.epochs,
        ws.cta_status,
        failure,
    )
    po = dict(
        POOL_TOKENS=k.numel() // 256,
        SPLITS=ws.splits,
        MAX_PAGES=ws.pages_per_chunk,
        SCALE=softmax_scale,
        LEFT=window_left,
        HAS_MAP=mapping is not None,
        MAP_PAGE=mapping_page_aligned,
        SKIP=skip,
        PIPELINED=ws.pages_per_chunk > 8,
        DIRECT_KV=direct_kv,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
        EARLY_PLAN=bool(EARLY_PLAN),
    )
    if heads is not None:
        # S1's q/k head norm + RoPE (and k-cache write) run in these CTAs; q is
        # the S1-out buffer this fills, heads = dict(p1, pos, cos, loc, k_out, eps, q_scale)
        po.update(
            FUSE_HEADS=True, P1=heads["p1"], POS=heads["pos"], COS=heads["cos"],
            LOC=heads["loc"], KOut=heads["k_out"], EPS=float(heads["eps"]),
            QSCALE=float(heads["q_scale"]), ROPE=heads["cos"] is not None,
            HEADS_MAP=mapping is not None,
        )
    ma = (ws.partial, ws.lse, ws.status, ws.cta_status, output, failure)
    mo = dict(
        SPLITS=ws.splits,
        Seed=delay_seed,
        CACHE=merge_cache,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if norm is not None:  # (gate, normalized): S3's head norm/gate fused into merge
        mo.update(Gate=norm[0], Norm=norm[1], FUSE_NORM=True)
    if EARLY_ISSUE:
        mo.update(EARLY=True)
    key = (
        window_left,
        mapping is not None,
        mapping_page_aligned,
        skip,
        delay_seed is not None,
        softmax_scale,
        direct_kv,
        merge_cache,
        norm is not None,
        heads is not None,
        bool(EARLY_PLAN),
    )
    if key not in ws.admitted:
        records = []
        for name, fn, aa, kk, grid in [
            ("partial", partials, pa, po, ws.ctas),
            ("merge", merge, ma, mo, 256),
        ]:
            compiled = fn.warmup(*aa, **kk, grid=(grid,))
            r = kernel_occupancy("split-S2-" + name, compiled)
            assert r.active_ctas_per_cu >= 1
            records.append(
                dict(
                    phase=name,
                    waves=compiled.metadata.num_warps,
                    vgprs=compiled.n_regs,
                    spills=compiled.n_spills,
                    lds=compiled.metadata.shared,
                    occupancy=r.active_ctas_per_cu,
                    grid=grid,
                )
            )
        ws.admitted[key] = records
    partials[(ws.ctas,)](*pa, **po)
    merge[(256,)](*ma, **mo)
    return output

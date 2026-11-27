"""T11 private S2 same-body decomposition; never selected by the chain."""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard,
    tlx_standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    BlockedEncoding,
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
            and (self.parent == other.parent)
        )

    def to_ir(self, builder, *_):
        return builder.get_slice_layout(self.dim, self.parent.to_ir(builder))


MMA = tl.constexpr(AttentionMma())
DOT_A = tl.constexpr(AttentionMma(0))
DOT_B = tl.constexpr(AttentionMma(1))
MERGE_PARENT = BlockedEncoding([1, 4], [2, 16], [4, 1], [1, 0])
MERGE = tl.constexpr(MERGE_PARENT)
MERGE_SPLIT = tl.constexpr(Slice(1, MERGE_PARENT))


@triton.jit
def fixed(value, layout: tl.constexpr):
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
    if HAS_MAP and (not MAP_PAGE):
        slot = tl.load(Map + slot)
    return (slot, (token < end) & (slot >= 0))


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
    MODE: tl.constexpr,
):
    if MODE == "compute":
        return offset + tl.arange(0, 64) < end
    slot, valid = slots_at(pages, Map, offset, start, end, MAX_PAGES, HAS_MAP, MAP_PAGE)
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
    MODE: tl.constexpr,
    LENGTH: tl.constexpr,
    DIRECT_KV: tl.constexpr = False,
):
    r = tl.arange(0, 64)
    d = tl.arange(0, 64)
    if MODE == "compute":
        q = (((r[:, None] + d[None, :] + head) % 8 - 4).to(tl.float32) * 0.0625).to(
            tl.bfloat16
        )
    else:
        q = tl.load(
            Q + r[:, None] // 8 * 2048 + (head * 8 + r[:, None] % 8) * 64 + d[None, :]
        )
    q = fixed(q, DOT_A)
    length = LENGTH if MODE == "compute" else tl.load(Length)
    first = tl.maximum(0, length - 8 - LEFT) // 64 * 64 if LEFT >= 0 else 0
    tiles = tl.cdiv(length - first, 64)
    start = first + split * tiles // SPLITS * 64
    end = tl.minimum(length, first + (split + 1) * tiles // SPLITS * 64)
    pages = (
        tl.zeros((MAX_PAGES,), tl.int32)
        if MODE == "compute"
        else cached_pages(Pages, Map, start, end, MAX_PAGES, MAP_PAGE)
    )
    acc = fixed(tl.zeros((64, 64), tl.float32), MMA)
    maximum = tl.full((64,), float("-inf"), tl.float32)
    denom = tl.zeros((64,), tl.float32)
    scale = SCALE * 1.4426950408889634 * (1.0 if MODE == "compute" else tl.load(KS))
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
    if MODE == "compute":
        init = (
            ((r[:, None] * 3 + d[None, :] + head + split) % 16 - 8).to(tl.float32)
            * 0.0625
        ).to(K.dtype.element_ty)
        for init_bank in tl.static_range(2):
            tlx.local_store(tlx.local_view(sk, init_bank), init)
            tlx.local_store(tlx.local_view(sv, init_bank), init)
        tlx.workgroup_barrier()
    score = fixed(tl.full((64, 64), float("-inf"), tl.float32), MMA)
    current_valid = tl.full((64,), False, tl.int1)
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
            MODE,
        )
        tlx.async_load_wait_group(0)
        tlx.workgroup_barrier()
        if MODE != "load":
            if DIRECT_KV:
                kk = fixed(
                    tlx.local_load(
                        tlx.local_trans(tlx.local_view(sk, 0)), layout=DOT_B
                    ),
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
    if MODE == "load":
        crc = tl.full((), 0, tl.uint32)
        load_iteration = 0
        for load_offset in range(start, end, 64):
            load_bank = load_iteration % 2
            if load_offset + 64 < end:
                copy_async_chunk(
                    K,
                    V,
                    pages,
                    Map,
                    load_offset + 64,
                    start,
                    end,
                    head,
                    MAX_PAGES,
                    HAS_MAP,
                    MAP_PAGE,
                    LEFT >= 0 or MAX_PAGES >= 64,
                    sk,
                    sv,
                    1 - load_bank,
                    MODE,
                )
            tlx.async_load_wait_group(0)
            tlx.workgroup_barrier()
            if DIRECT_KV:
                ka = fixed(
                    tlx.local_load(
                        tlx.local_trans(tlx.local_view(sk, load_bank)), layout=DOT_B
                    ),
                    DOT_B,
                )
            else:
                ka = tlx.local_load(tlx.local_view(sk, load_bank), layout=CHUNK_LOAD)
            if DIRECT_KV:
                va = fixed(
                    tlx.local_load(tlx.local_view(sv, load_bank), layout=DOT_B), DOT_B
                )
            else:
                va = tlx.local_load(tlx.local_view(sv, load_bank), layout=CHUNK_LOAD)
            selected = (r[:, None] == 0) & (d[None, :] < 16)
            selected_k = (
                (d[None, :] == 0) & (r[:, None] < 16) if DIRECT_KV else selected
            )
            crc += tl.sum(
                tl.where(selected_k, ka.to(tl.uint8, bitcast=True).to(tl.uint32), 0)
            ) + tl.sum(
                tl.where(selected, va.to(tl.uint8, bitcast=True).to(tl.uint32), 0)
            )
            tlx.workgroup_barrier()
            load_iteration += 1
        return (crc, tl.full((), 0, tl.float32))
    else:
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
                    MODE,
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
                    kk = tlx.local_load(
                        tlx.local_view(sk, 1 - bank), layout=CHUNK_LOAD
                    ).to(tl.bfloat16)
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
        value = (
            acc
            * (
                (1.0 if MODE == "compute" else tl.load(VS))
                / tl.where(denom > 0, denom, 1.0)
            )[:, None]
        )
        lse = tl.where(denom > 0, maximum + tl.log2(denom), float("-inf"))
        return (value, lse)


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
    MODE: tl.constexpr = "full",
    LENGTH: tl.constexpr = 0,
    Checksum=None,
    DIRECT_KV: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if MODE != "compute":
        tag = tlx_standin.advance(Epochs, pid)
    allowed = (
        tl.full((), True, tl.int1)
        if MODE == "compute"
        else tlx_guard.enabled(Failure, 2)
    )
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
            MODE,
            LENGTH,
            DIRECT_KV=DIRECT_KV,
        )
        if MODE == "load":
            tl.store(Checksum + pid, value)
        else:
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

"""TLX S1: exact SK8 projection and pinned-load/head reductions; normal stores.

No output-store index/value pins. Preserve the Gluon head arithmetic, rounding,
rotary pairing and exact cache writes. Padded inactive rows are zero, not copied
from row7; every observable write is masked to the eight valid rows.
"""

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_code_warm,
    tlx_guard,
    tlx_memory,
    tlx_standin,
    tlx_workspace,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_asm import (
    _load_asm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    BlockedEncoding,
    require_pin_support,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_projection

_make_common_workspace = tlx_workspace.make_s1
HEAD = tl.constexpr(BlockedEncoding([1, 2], [1, 32], [4, 1], [1, 0]))


@triton.jit
def _head_add(a, b):
    return a + b


@triton.jit
def norm_rope(
    value,
    POS,
    COS,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    ROPE: tl.constexpr,
    QUERY: tl.constexpr,
    L: tl.constexpr = HEAD,
    TOK: tl.constexpr = 1,
    OFF: tl.constexpr = 0,
):
    """Rows are (token, head) pairs, token = row // TOK; only tokens < 8 are real.
    OFF > 0 builds the index ranges from distinct ops (CSE would otherwise merge
    them with a caller's own pinned tl.arange(0, 64))."""
    row = tl.arange(OFF, OFF + value.shape[0]) - OFF
    tok = row // TOK
    d = tl.arange(OFF, OFF + 64) - OFF
    value = tlx.require_layout(value, L, pin=True)
    squares = tlx.require_layout(value * value, L, pin=True)
    if OFF > 0:
        # same tt.reduce tree as tl.sum, but its own callee: a shared
        # sum__fp32S64_64S function would carry this layout into the caller's sums
        total = tl.reduce(squares, 1, _head_add)
    else:
        total = tl.sum(squares, 1)
    inv = tl.rsqrt(total / 64 + EPS)
    value = (value * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    if QUERY:
        value = (value * SCALE).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        value = tlx.require_layout(value, L, pin=True)
        pos = tl.load(POS + tok, tok < 8, 0)
        cosine = (
            tl.load(COS + pos[:, None] * 64 + d[None, :] // 2, tok[:, None] < 8, 0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        sine = (
            tl.load(COS + pos[:, None] * 64 + 32 + d[None, :] // 2, tok[:, None] < 8, 0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        # Pin only rotary compute inputs, never output-store indices.
        cosine = tlx.require_layout(cosine, L, pin=True)
        sine = tlx.require_layout(sine, L, pin=True)
        index = tlx.require_layout(
            tl.broadcast_to(d[None, :] ^ 1, (value.shape[0], 64)), L, pin=True
        )
        other = tl.gather(value, index, 1)
        even = tlx.require_layout(
            tl.broadcast_to(d[None, :] % 2 == 0, (value.shape[0], 64)), HEAD
        )
        rotated = tl.where(even, -other, other)
        first = (value * cosine).to(tl.bfloat16).to(tl.float32)
        second = (rotated * sine).to(tl.bfloat16).to(tl.float32)
        value = first + second
    return value.to(tl.bfloat16)


# Two-wave shapes for the fused projection + heads kernel (projection runs 2 waves).
HEAD2 = tl.constexpr(BlockedEncoding([1, 2], [1, 32], [2, 1], [1, 0]))
R64 = tl.constexpr(BlockedEncoding([1], [32], [2], [0]))


@triton.jit
def load_head_scoped(P, head):
    """[8, 64] words of one head, SCOPE_DEV b128 loads (two per lane), one wait."""
    lane = tlx.require_layout(tl.arange(0, 64), R64)
    base = P + (lane // 8) * 4608 + head * 64 + (lane % 8) * 8
    spec: tl.constexpr = _load_asm(2, 1)
    raw = tl.inline_asm_elementwise(
        spec[0], spec[1],
        [tlx.release_layout(base), tlx.release_layout(base + 4),
         tlx.release_layout(lane * 0 + 1)],
        dtype=(tl.int32,) * 9, is_pure=False, pack=1,
    )
    c = tl.arange(0, 8)[None, :]
    v = tl.zeros((64, 8), tl.int32)
    for k in tl.static_range(8):
        v = tl.where(c == k, raw[k][:, None], v)
    return tl.reshape(v, (8, 64))


@triton.jit
def project_vg(
    X, W, P, Status, LOC, MAP, VC, V, G, VS, Failure,
    HAS_MAP: tl.constexpr, SKIP: tl.constexpr = -1, Seed=None, EARLY: tl.constexpr = False,
    DEPTH: tl.constexpr = 4,
    COPY: tl.constexpr = "tdm",
    BN: tl.constexpr = 16,
):
    """S1 projection whose v and gate tiles also write V, the v cache and G
    (no norm needed there). q/k norm+RoPE run in S2 (tlx_attention FUSE_HEADS).
    BN columns per CTA (16 or 32); a 64-wide head spans 64 // BN tiles."""
    tile = tl.program_id(0)
    ok, value = tlx_projection.project_body(
        X, W, P, Status, True, True, Failure, SKIP, Seed, None,
        4608, 8192, 1, 8, True, DEPTH, COPY, False, EARLY, BN,
    )
    head = tile // (64 // BN)
    if ok & (head >= 36):
        r = tl.arange(0, 16)
        c = (tile % (64 // BN)) * BN + tl.arange(0, BN)
        if head < 40:
            kh = head - 36
            kv = value.to(tl.bfloat16)
            tl.store(V + r[:, None] * 256 + kh * 64 + c[None, :], kv, r[:, None] < 8)
            slot = tl.load(LOC + r, r < 8, -1).to(tl.int64)
            if HAS_MAP:
                slot = tl.load(MAP + slot, (r < 8) & (slot >= 0), -1).to(tl.int64)
            tl.store(
                VC + slot[:, None] * 256 + kh * 64 + c[None, :],
                kv.to(tl.float32) / tl.load(VS),
                (r[:, None] < 8) & (slot[:, None] >= 0),
            )
        else:
            tl.store(G + r[:, None] * 2048 + (head - 40) * 64 + c[None, :],
                     value.to(tl.bfloat16), r[:, None] < 8)


@triton.jit
def project_heads(
    X, W, P, Status, State, POS, COS, LOC, MAP, KC, VC, Q, K, V, G, KS, VS, Failure,
    EPS: tl.constexpr, SCALE: tl.constexpr, ROPE: tl.constexpr, HAS_MAP: tl.constexpr,
    SKIP: tl.constexpr = -1, Seed=None,
    CodeSink=None, code_off=0, code_quads=0, WARM: tl.constexpr = False,
    Trace=None,
    PUBLISH: tl.constexpr = 1,
):
    """S1 in one launch: 288 projection tiles; the last of each head's 4 tiles
    (relaxed counter, no polling) runs that head's norm/RoPE/KV write/gate."""
    tile = tl.program_id(0)
    t0 = tlx_standin.clock()
    if WARM:
        if tile >= 224:
            tlx_code_warm.code_warm(tile - 224, 64, CodeSink, code_off, code_quads)
    ok, _ = tlx_projection.project_body(
        X, W, P, Status, True, True, Failure, SKIP, Seed, None,
        4608, 8192, 1, 8, True, 4, "tdm", PUBLISH,
    )
    t0 = tlx_standin.clock()
    head = tile // 4
    lane = tlx.require_layout(tl.arange(0, 64), R64)
    if PUBLISH == 2:
        sgone = tl.atomic_xchg(Status + tile + lane * 0, lane * 0 + ok.to(tl.int32),
                               mask=lane == 0, sem="relaxed", scope="gpu")
        ok = ok & (tl.sum(sgone * 0, 0) == 0)
    else:
        tlx_memory.store_words(Status + tile + lane * 0, lane * 0 + ok.to(tl.int32), lane == 0)
        tlx_memory.drain_stores()
    t1 = tlx_standin.clock()
    old = tl.atomic_add(State + head * 32 + lane * 0, ok.to(tl.int32) * 0 + 1, mask=lane == 0,
                        sem="relaxed", scope="gpu")
    old = tl.sum(tl.where(lane == 0, old, 0), 0)
    t2 = tlx_standin.clock()
    t3 = t2
    if old % 4 == 3:
        st = tlx_memory.load_words(Status + head * 4 + lane % 4)
        all_ok = tl.min(st, 0) == 1
        if all_ok:
            heads_body(head, P, POS, COS, LOC, MAP, KC, VC, Q, K, V, G, KS, VS,
                       EPS, SCALE, ROPE, HAS_MAP, 8, HEAD2, True)
        tlx_guard.finish(Failure, all_ok, head, 1)
        t3 = tlx_standin.clock()
    if Trace is not None:
        kq = tl.arange(0, 4)
        tl.store(Trace + tile * 4 + kq,
                 tl.where(kq == 0, t0, tl.where(kq == 1, t1, tl.where(kq == 2, t2, t3))).to(tl.int64))


@triton.jit
def heads_body(head, P, POS, COS, LOC, MAP, KC, VC, Q, K, V, G, KS, VS,
               EPS: tl.constexpr, SCALE: tl.constexpr, ROPE: tl.constexpr,
               HAS_MAP: tl.constexpr, HEAD_ROWS: tl.constexpr, L: tl.constexpr,
               SCOPED: tl.constexpr):
    """One 64-column head of the projection: q/k norm+RoPE (+KV write), v, gate."""
    row = tl.arange(0, HEAD_ROWS)
    d = tl.arange(0, 64)
    if SCOPED:
        value = load_head_scoped(P, head)
    else:
        value = tl.load(
            P + row[:, None] * 4608 + head * 64 + d[None, :], row[:, None] < 8, 0
        )
    value = value.to(tl.float32, bitcast=True)
    value = tlx.require_layout(value, L, pin=True)
    if head < 32:
        query = norm_rope(value, POS, COS, EPS, SCALE, ROPE, True, L)
        tl.store(
            Q + row[:, None] * 2048 + head * 64 + d[None, :],
            query,
            row[:, None] < 8,
        )
    elif head < 40:
        if head < 36:
            kh = head - 32
            kv = norm_rope(value, POS, COS, EPS, 1.0, ROPE, False, L)
            tl.store(
                K + row[:, None] * 256 + kh * 64 + d[None, :], kv, row[:, None] < 8
            )
            cache = KC
            scale = tl.load(KS)
        else:
            kh = head - 36
            kv = value.to(tl.bfloat16)
            tl.store(
                V + row[:, None] * 256 + kh * 64 + d[None, :], kv, row[:, None] < 8
            )
            cache = VC
            scale = tl.load(VS)
        slot = tl.load(LOC + row, row < 8, -1).to(tl.int64)
        if HAS_MAP:
            slot = tl.load(MAP + slot, (row < 8) & (slot >= 0), -1).to(tl.int64)
        tl.store(
            cache + slot[:, None] * 256 + kh * 64 + d[None, :],
            kv.to(tl.float32) / scale,
            (row[:, None] < 8) & (slot[:, None] >= 0),
        )
    else:
        tl.store(
            G + row[:, None] * 2048 + (head - 40) * 64 + d[None, :],
            value.to(tl.bfloat16),
            row[:, None] < 8,
        )


@triton.jit
def qkvg_heads(
    P,
    State,
    Status,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    Q,
    K,
    V,
    G,
    KS,
    VS,
    Failure,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_MAP: tl.constexpr,
    HEAD_ROWS: tl.constexpr = 16,
    Seed=None,
):
    head = tl.program_id(0)
    tag = tlx_standin.advance(State, head)
    j = tl.arange(0, 4)
    ok = tl.min(tl.load(Status + head * 4 + j), 0) == 1
    tlx_guard.delay(Seed, head, 1)
    if ok:
        heads_body(head, P, POS, COS, LOC, MAP, KC, VC, Q, K, V, G, KS, VS,
                   EPS, SCALE, ROPE, HAS_MAP, HEAD_ROWS, HEAD, False)
        tl.store(State + 288 * 32 + head, tag)
    tlx_guard.finish(Failure, ok, head, 1)


def make_workspace(device, *, ctas=None, shape=None):
    assert ctas is None or ctas == 288
    return _make_common_workspace(device, ctas=288, shape=shape)


def qkvg(
    x,
    weight,
    positions,
    cos_sin,
    cache_k,
    cache_v,
    locations,
    mapping,
    eps,
    q_scale,
    k_scale,
    v_scale,
    *,
    ctas=None,
    workspace=None,
    shape=None,
    skip=-1,
    delay_seed=None,
    trace=None,
    failure=None,
    guard_mode=0,
    head_rows=16,
    fused=False,
):
    assert guard_mode == 0 and (trace is None or fused)
    assert head_rows in (8, 16)
    require_pin_support()
    ws = (
        make_workspace(x.device, ctas=ctas, shape=shape)
        if workspace is None
        else workspace
    )
    assert ws.ctas == 288 and (ctas is None or ctas == 288)
    q = torch.empty((8, 2048), device=x.device, dtype=torch.bfloat16)
    k = torch.empty((8, 256), device=x.device, dtype=torch.bfloat16)
    v, gout = torch.empty_like(k), torch.empty_like(q)
    pa = (x, weight, ws.projection, ws.status)
    po = dict(
        Failure=failure,
        N=4608,
        K=8192,
        STAGE=1,
        SK=8,
        SKIP=skip,
        Seed=delay_seed,
        num_warps=2,
        enable_fp_fusion=False,
    )
    ha = (
        ws.projection,
        ws.state,
        ws.status,
        positions,
        cos_sin,
        locations,
        mapping,
        cache_k,
        cache_v,
        q,
        k,
        v,
        gout,
        k_scale,
        v_scale,
        failure,
    )
    ho = dict(
        EPS=float(eps),
        SCALE=float(q_scale),
        ROPE=cos_sin is not None,
        HAS_MAP=mapping is not None,
        HEAD_ROWS=head_rows,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
    )
    key = (
        head_rows,
        skip,
        delay_seed is not None,
        cos_sin is not None,
        mapping is not None,
        failure is not None,
        float(eps),
        float(q_scale),
    )
    if key not in ws.admitted:
        records = []
        for name, fn, aa, kk, grid in [
            ("projection", tlx_projection.project, pa, po, 288),
            ("heads", qkvg_heads, ha, ho, 72),
        ]:
            kernel = fn.warmup(*aa, **kk, grid=(grid,))
            r = kernel_occupancy("independent-S1-" + name, kernel)
            assert r.active_ctas_per_cu >= 1 and kernel.metadata.shared <= 163840
            records.append(
                dict(
                    phase=name,
                    waves=kernel.metadata.num_warps,
                    vgprs=kernel.n_regs,
                    spills=kernel.n_spills,
                    lds=kernel.metadata.shared,
                    occupancy=r.active_ctas_per_cu,
                    grid=grid,
                )
            )
        ws.admitted[key] = records
    if fused == "vg":
        project_vg[(4608 // tlx_projection.BN_S1,)](
            x, weight, ws.projection, ws.status, locations, mapping, cache_v, v, gout,
            v_scale, failure, HAS_MAP=mapping is not None, SKIP=skip, Seed=delay_seed,
            EARLY=tlx_projection.EARLY_ISSUE, DEPTH=tlx_projection.DEPTH_S1,
            COPY=tlx_projection.COPY_MODE, BN=tlx_projection.BN_S1,
            num_warps=2, enable_fp_fusion=False,
        )
        return q, k, v, gout
    if fused:
        fo = dict(ho)
        del fo["HEAD_ROWS"]
        fo.update(SKIP=skip, num_warps=2)
        fa = (x, weight, ws.projection, ws.status) + ha[1:2] + ha[3:]
        if fused == "warm":
            code = ws.admitted.get("code_warm")
            if code is None:
                sink = torch.zeros(16, device=x.device, dtype=torch.int32)
                k0 = project_heads.warmup(*fa, **fo, CodeSink=sink, code_off=0,
                                          code_quads=0, WARM=True, grid=(288,))
                code = (sink, tlx_code_warm.text_range(k0), k0)
                ws.admitted["code_warm"] = code
            fo.update(CodeSink=code[0], code_off=code[1][0], code_quads=code[1][1], WARM=True)
        if trace is not None:
            fo.update(Trace=trace)
        if fused == "xchg":
            fo.update(PUBLISH=2)
        project_heads[(288,)](*fa, **fo)
        return q, k, v, gout
    tlx_projection.project[(288,)](*pa, **po)
    qkvg_heads[(72,)](*ha, **ho)
    return q, k, v, gout

"""Single-launch TP4 S1 on the shared M8 BF16 core; GPU-local handoffs only.

Each CTA computes32 columns. For Q/K, the even CTA of each pair consumes
its complete64-column head; V/G store directly without a handoff. Each Q/K
consumer polls its own128-byte flag line with an exact launch tag, a4096-poll
cap and a per-CU cycle check every64 polls. Workspace launches must be serial
on one stream. Below144 CTAs, each CTA computes both halves of its heads.
Check the workspace's final72 status words after execution; discard state
following any timeout. Allocate and zero workspace before capture/timing.
"""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    store_bf16_tile,
    store_fp8_tile,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import (
    load_input,
    skinny_tile,
)


class S1Workspace(msgspec.Struct):
    projection: torch.Tensor
    state: torch.Tensor
    ctas: int = 192


def make_workspace(device, *, ctas=192):
    """Allocate reusable single-stream scratch; zero state once outside graph capture."""
    return S1Workspace(
        torch.empty((8, 4608), device=device, dtype=torch.int32),
        torch.zeros(216 * 32 + 72, device=device, dtype=torch.int32),
        ctas,
    )


@g.jit
def _store_half(P, task, value):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [gl.num_warps(), 1], [1, 0]
    )
    row = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    col = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    value = (
        gl.convert_layout(value, layout)
        .to(gl.bfloat16)
        .to(gl.float32)
        .to(gl.int32, bitcast=True)
    )
    scoped.store_words(
        P + row[:, None] * 4608 + task * 32 + col[None, :], value, row[:, None] < 8
    )


@g.jit
def _norm_rope(
    value,
    POS,
    COS,
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    QUERY: gl.constexpr,
    RowStart=0,
    PAD_ROWS: gl.constexpr = 16,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [1, 32], [gl.num_warps(), 1], [1, 0]
    )
    row = gl.arange(0, PAD_ROWS, layout=gl.SliceLayout(1, layout)) + RowStart
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    inv = gl.rsqrt(gl.sum(value * value, 1) / 64 + EPS)
    value = (value * inv[:, None]).to(gl.bfloat16).to(gl.float32)
    if QUERY:
        value = (value * SCALE).to(gl.bfloat16).to(gl.float32)
    if ROPE:
        pos = gl.load(POS + row, row < 8, other=0)
        cosine = (
            gl.load(
                COS + pos[:, None] * 64 + d[None, :] // 2, row[:, None] < 8, other=0
            )
            .to(gl.bfloat16)
            .to(gl.float32)
        )
        sine = (
            gl.load(
                COS + pos[:, None] * 64 + 32 + d[None, :] // 2,
                row[:, None] < 8,
                other=0,
            )
            .to(gl.bfloat16)
            .to(gl.float32)
        )
        index = gl.full((PAD_ROWS, 64), 0, gl.int32, layout) + (d[None, :] ^ 1)
        other = gl.gather(value, index, 1)
        rotated = gl.where(d[None, :] % 2 == 0, -other, other)
        first = (value * cosine).to(gl.bfloat16).to(gl.float32)
        second = (rotated * sine).to(gl.bfloat16).to(gl.float32)
        value = first + second
    return value.to(gl.bfloat16)


@g.jit
def head_epilogue(
    P,
    head,
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
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
    Value=None,
    SCOPED_IO: gl.constexpr = False,
    RowStart=0,
    PAD_ROWS: gl.constexpr = 16,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [1, 32], [gl.num_warps(), 1], [1, 0]
    )
    row = gl.arange(0, PAD_ROWS, layout=gl.SliceLayout(1, layout)) + RowStart
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    if Value is None:
        value = scoped.load_words_masked(
            P + row[:, None] * 4608 + head * 64 + d[None, :], row[:, None] < 8
        ).to(gl.float32, bitcast=True)
    else:
        small = gl.convert_layout(Value, layout)
        index = gl.minimum(row[:, None], 7) + gl.full(
            (PAD_ROWS, 64), 0, gl.int32, layout
        )
        value = gl.gather(small, index, 0)
    if head < 32:
        q_value = _norm_rope(
            value, POS, COS, EPS, SCALE, ROPE, True, RowStart, PAD_ROWS
        )
        if SCOPED_IO:
            store_bf16_tile(
                Q + row[:, None] * 2048 + head * 64 + d[None, :],
                q_value,
                row[:, None] < 8,
                PAD_ROWS,
                64,
            )
        else:
            gl.store(
                Q + row[:, None] * 2048 + head * 64 + d[None, :],
                q_value,
                row[:, None] < 8,
            )
    elif head < 40:
        if head < 36:
            kh = head - 32
            kv_value = _norm_rope(
                value, POS, COS, EPS, 1.0, ROPE, False, RowStart, PAD_ROWS
            )
            if SCOPED_IO:
                store_bf16_tile(
                    K + row[:, None] * 256 + kh * 64 + d[None, :],
                    kv_value,
                    row[:, None] < 8,
                    PAD_ROWS,
                    64,
                )
            else:
                gl.store(
                    K + row[:, None] * 256 + kh * 64 + d[None, :],
                    kv_value,
                    row[:, None] < 8,
                )
            cache = KC
            scale = gl.load(KS)
        else:
            kh = head - 36
            kv_value = value.to(gl.bfloat16)
            if SCOPED_IO:
                store_bf16_tile(
                    V + row[:, None] * 256 + kh * 64 + d[None, :],
                    kv_value,
                    row[:, None] < 8,
                    PAD_ROWS,
                    64,
                )
            else:
                gl.store(
                    V + row[:, None] * 256 + kh * 64 + d[None, :],
                    kv_value,
                    row[:, None] < 8,
                )
            cache = VC
            scale = gl.load(VS)
        slot = gl.load(LOC + row, row < 8, other=-1).to(gl.int64)
        if HAS_MAP:
            slot = gl.load(MAP + slot, (row < 8) & (slot >= 0), other=-1).to(gl.int64)
        if SCOPED_IO:
            store_fp8_tile(
                cache + slot[:, None] * 256 + kh * 64 + d[None, :],
                (kv_value.to(gl.float32) / scale).to(cache.dtype.element_ty),
                (row[:, None] < 8) & (slot[:, None] >= 0),
                PAD_ROWS * 64,
            )
        else:
            gl.store(
                cache + slot[:, None] * 256 + kh * 64 + d[None, :],
                kv_value.to(gl.float32) / scale,
                (row[:, None] < 8) & (slot[:, None] >= 0),
            )

    else:
        if SCOPED_IO:
            store_bf16_tile(
                G + row[:, None] * 2048 + (head - 40) * 64 + d[None, :],
                value.to(gl.bfloat16),
                row[:, None] < 8,
                PAD_ROWS,
                64,
            )
        else:
            gl.store(
                G + row[:, None] * 2048 + (head - 40) * 64 + d[None, :],
                value.to(gl.bfloat16),
                row[:, None] < 8,
            )


@g.jit
def _direct_half(task, value, LOC, MAP, VC, V, G, VS, HAS_MAP: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [1, 32], [gl.num_warps(), 1], [1, 0]
    )
    row = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    d = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    value = gl.convert_layout(value, layout).to(gl.bfloat16)
    if task < 80:
        kh = (task - 72) // 2
        feature = (task % 2) * 32 + d
        gl.store(
            V + row[:, None] * 256 + kh * 64 + feature[None, :], value, row[:, None] < 8
        )
        slot = gl.load(LOC + row, row < 8, other=-1).to(gl.int64)
        if HAS_MAP:
            slot = gl.load(MAP + slot, (row < 8) & (slot >= 0), other=-1).to(gl.int64)
        gl.store(
            VC + slot[:, None] * 256 + kh * 64 + feature[None, :],
            value.to(gl.float32) / gl.load(VS),
            (row[:, None] < 8) & (slot[:, None] >= 0),
        )
    else:
        gl.store(
            G + row[:, None] * 2048 + (task - 80) * 32 + d[None, :],
            value,
            row[:, None] < 8,
        )


@g.jit
def _advance_tag(State, task):
    lane = gl.arange(
        0,
        gl.num_warps() * 32,
        layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]),
    )
    old = gl.atomic_add(
        State + task * 32 + gl.full_like(lane, 0),
        1,
        mask=lane == 0,
        sem="relaxed",
        scope="gpu",
    )
    return gl.sum(gl.where(lane == 0, old + 1, 0), 0)


@g.jit
def s1_stage(
    task_index,
    ctas,
    X,
    W,
    P,
    State,
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
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
    PAIRED: gl.constexpr,
):
    if PAIRED:
        if task_index < 144:
            tag = task_index * 0
            if task_index < 72:
                tag = _advance_tag(State, task_index)
            cached_x = load_input(X, 8192, 8, 64)
            value = skinny_tile(
                W, cached_x, task_index, 4608, 8192, 32, 64, 8, 8, 2, ".cs", False
            )
            head = task_index // 2
            if task_index >= 72:
                _direct_half(task_index, value, LOC, MAP, VC, V, G, VS, HAS_MAP)
                if task_index % 2 == 0:
                    gl.store(State + 216 * 32 + head, 1)
            else:
                _store_half(P, task_index, value)
                flag = State + 144 * 32 + head * 32
                if task_index % 2 == 1:
                    scoped.publish(flag, tag)
                else:
                    scoped.drain_stores()
                    seen, polls = scoped.wait_flag(
                        flag,
                        tag,
                        100_000_000,
                        MAX_SPINS=4096,
                        CLOCK_EVERY=64,
                        SHADER_CLOCK=True,
                    )
                    gl.store(State + 216 * 32 + head, (seen == tag).to(gl.int32))
                    if seen == tag:
                        head_epilogue(
                            P,
                            head,
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
                            EPS,
                            SCALE,
                            ROPE,
                            HAS_MAP,
                        )
    else:
        if task_index < 72:
            cached_x = load_input(X, 8192, 8, 64)
            for head in range(task_index, 72, ctas):
                first = skinny_tile(
                    W, cached_x, head * 2, 4608, 8192, 32, 64, 8, 8, 2, ".cs", False
                )
                _store_half(P, head * 2, first)
                second = skinny_tile(
                    W, cached_x, head * 2 + 1, 4608, 8192, 32, 64, 8, 8, 2, ".cs", False
                )
                _store_half(P, head * 2 + 1, second)
                scoped.drain_stores()
                head_epilogue(
                    P,
                    head,
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
                    EPS,
                    SCALE,
                    ROPE,
                    HAS_MAP,
                )
                gl.store(State + 216 * 32 + head, 1)


@g.jit
def fused_s1(
    X,
    W,
    P,
    State,
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
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
    PAIRED: gl.constexpr,
):
    s1_stage(
        gl.program_id(0),
        gl.num_programs(0),
        X,
        W,
        P,
        State,
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
        EPS,
        SCALE,
        ROPE,
        HAS_MAP,
        PAIRED,
    )


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
    ctas=192,
    workspace=None,
):
    """Run S1 in one compute launch; reuse workspace for graph capture and timing."""
    assert x.shape == (8, 8192) and weight.shape == (4608, 8192)
    assert (
        x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
    )
    assert cache_k.dtype == cache_v.dtype == torch.float8_e4m3fn
    assert cache_k.shape == cache_v.shape and cache_k.shape[-3:] == (16, 4, 64)
    assert cache_k.is_contiguous() and cache_v.is_contiguous()
    assert positions.shape == locations.shape == (8,)
    assert 0 < ctas <= torch.cuda.get_device_properties(x.device).multi_processor_count
    assert all(
        t.device == x.device
        for t in (weight, positions, cache_k, cache_v, locations, k_scale, v_scale)
    )
    assert cos_sin is None or (
        cos_sin.device == x.device
        and cos_sin.is_contiguous()
        and cos_sin.shape[1] == 64
    )
    assert mapping is None or (mapping.device == x.device and mapping.is_contiguous())
    if workspace is None:
        workspace = make_workspace(x.device, ctas=ctas)
    assert workspace.ctas == ctas
    assert (
        workspace.projection.shape == (8, 4608)
        and workspace.projection.dtype == torch.int32
    )
    assert (
        workspace.state.shape == (216 * 32 + 72,)
        and workspace.state.dtype == torch.int32
    )
    assert workspace.projection.is_contiguous() and workspace.state.is_contiguous()
    assert workspace.projection.device == workspace.state.device == x.device
    q = torch.empty((8, 2048), device=x.device, dtype=torch.bfloat16)
    k = torch.empty((8, 256), device=x.device, dtype=torch.bfloat16)
    v, gate = torch.empty_like(k), torch.empty_like(q)
    fused_s1[(ctas,)](
        x,
        weight,
        workspace.projection,
        workspace.state,
        positions,
        cos_sin,
        locations,
        mapping,
        cache_k,
        cache_v,
        q,
        k,
        v,
        gate,
        k_scale,
        v_scale,
        float(eps),
        float(q_scale),
        cos_sin is not None,
        mapping is not None,
        ctas >= 144,
        num_warps=16,
    )
    return q, k, v, gate

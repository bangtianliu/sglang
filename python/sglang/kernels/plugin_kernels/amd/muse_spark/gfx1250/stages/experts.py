"""S5 W13/SwiGLU/W2 on native A16W4 weights; one GPU, one rank.

Outputs are rounded BF16 values held in FP32 scratch [expert, token, column].
All eight unique token rows are evaluated per present expert, which also covers
synthetic repeated slots for the same token/expert without dropping hits.

The static expert/tile schedule has two sweeps: each CTA produces its W13
work, then consumes its W2 work through private per-expert/tile flags. There is
no grid barrier. A consumer may wait for another CTA finishing W13, but cannot
prevent that producer from making progress. At least 128 resident CTAs are
required by the standalone policy. Polls have iteration and clock bounds.

State belongs to serial launches on one stream. Exact epoch flags reject stale
values. Arrival counts advance only for present experts and publish at every
complete set of 64 tiles. Discard all state/output following a false status.
Performance remains WIP; this is not connected to the runtime forward path.
"""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna5, gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.vector_memory import (
    join4,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attention import (
    _load128,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    store_projection,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import (
    _input,
    _mma,
    load_input,
)


@g.jit
def _fp4_raw(
    W,
    Scale,
    expert,
    tile,
    step,
    ROWS: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    S: gl.constexpr,
    K: gl.constexpr = 4096,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 16, 1], [1, 8, 4], [S, 1, gl.num_warps() // S], [1, 2, 0]
    )
    ss = gl.arange(0, S, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    kk = gl.arange(0, BK // 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    nn = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    offset = ss[:, None, None] * (K // S // 2) + step * (BK // 2) + kk[None, :, None]
    row = tile * BN + nn
    packed = gl.load(
        W + expert * ROWS * (K // 2) + row[None, None, :] * (K // 2) + offset,
        cache_modifier=".cs",
    )
    scales = gl.load(
        Scale
        + expert * ROWS * (K // 32)
        + row[None, None, :] * (K // 32)
        + offset // 16
    )
    return packed, scales


@g.jit
def _nibble(n):
    e = (n >> 1) & 3
    m = n & 1
    value = gl.where(
        e == 0,
        m.to(gl.float32) * 0.5,
        (1.0 + m.to(gl.float32) * 0.5) * gl.exp2(e.to(gl.float32) - 1.0),
    )
    return gl.where((n & 8) != 0, -value, value)


@g.jit
def _fp4_upcast(
    packed,
    scales,
    BN: gl.constexpr,
    BK: gl.constexpr,
    S: gl.constexpr,
    MODE: gl.constexpr,
):
    """Hardware packed conversion, with ALU and unit-scale diagnostic variants."""
    mma: gl.constexpr = _mma(gl.num_warps(), BN)
    operand: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    expanded = gl.reshape(gl.join(scales, scales).permute((0, 1, 3, 2)), (S, BK, BN))
    expanded = gl.convert_layout(expanded, operand)
    if MODE == "alu":
        low = _nibble(packed & 15)
        high = _nibble(packed >> 4)
        value = gl.reshape(gl.join(low, high).permute((0, 1, 3, 2)), (S, BK, BN))
        value = gl.convert_layout(value.to(gl.bfloat16), operand)
    else:
        raw = gl.convert_layout(packed, gl.DotOperandLayout(1, mma, 4))
        unit = (
            expanded
            if MODE == "direct"
            else gl.full((S, BK, BN), 127, gl.uint8, operand)
        )
        value = cdna5.scaled_upcast(raw, unit, gl.bfloat16, axis=1)
        value = gl.convert_layout(value, operand)
    if MODE != "direct":
        bits = gl.where(expanded == 0, 64, expanded.to(gl.uint16) << 7).to(gl.uint16)
        value = value * bits.to(gl.bfloat16, bitcast=True)
    return value


@g.jit
def _fp4_gemm(
    W,
    Scale,
    cached_x,
    expert,
    tile,
    ROWS: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    MODE: gl.constexpr,
):
    S: gl.constexpr = gl.num_warps() // (BN // 16)
    mma: gl.constexpr = _mma(gl.num_warps(), BN)
    acc = gl.full((S, 16, BN), 0, gl.float32, mma)
    p0, s0 = _fp4_raw(W, Scale, expert, tile, 0, ROWS, BN, BK, S)
    p1, s1 = _fp4_raw(W, Scale, expert, tile, 1, ROWS, BN, BK, S)
    for step in range(4096 // S // BK):
        a = _input(cached_x, step, 4096, BK, S, mma)
        b = _fp4_upcast(p0, s0, BN, BK, S, MODE)
        acc = gfx1250.wmma(a, b, acc)
        p0 = p1
        s0 = s1
        if step + 2 < 4096 // S // BK:
            p1, s1 = _fp4_raw(W, Scale, expert, tile, step + 2, ROWS, BN, BK, S)
    return gl.sum(acc, 0)


@g.jit
def fp4_tile(
    W,
    Scale,
    cached_x,
    expert,
    tile,
    ROWS: gl.constexpr,
    BN: gl.constexpr = 64,
    BK: gl.constexpr = 64,
    MODE: gl.constexpr = "direct",
    PAIR: gl.constexpr = False,
):
    value = _fp4_gemm(W, Scale, cached_x, expert, tile, ROWS, BN, BK, MODE)
    if PAIR:
        up = _fp4_gemm(
            W, Scale, cached_x, expert, tile + 4096 // BN, ROWS, BN, BK, MODE
        )
        value = value / (1.0 + gl.exp(-value)) * up
    return value


@g.jit
def _cached_inter(Inter, expert):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 4], [1, 0])
    q = gl.arange(0, 8192, layout=gl.BlockedLayout([1], [32], [16], [0]))
    a, b, c, d = _load128(Inter + expert * 8 * 4096 + q * 4, 16)
    value = join4(a, b, c, d, 8192).to(gl.float32, bitcast=True).to(gl.bfloat16)
    value = value.reshape((8, 4, 1024)).permute((1, 0, 2)).reshape((4, 8192))
    value = gl.convert_layout(value, layout)
    shared = gl.allocate_shared_memory(
        gl.bfloat16, [4, 8192], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    shared.store(value)
    return shared


@g.jit
def expert_stage(
    task_index,
    ctas,
    X,
    Ids,
    W13,
    S13,
    W2,
    S2,
    Inter,
    Y,
    Epoch,
    Arrivals,
    Flags,
    Status,
    SKIP: gl.constexpr = -1,
    MODE: gl.constexpr = "direct",
):
    gl.static_assert(gl.num_warps() == 16)
    tag = _advance_tag(Epoch, task_index)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 4], [1, 0])
    e = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    slot = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    ids = gl.load(Ids + slot)
    present = gl.max((e[:, None] == ids[None, :]).to(gl.int32), 1)
    present = gl.convert_layout(present, gl.BlockedLayout([1], [32], [16], [0]))
    e = gl.arange(0, 64, layout=gl.BlockedLayout([1], [32], [16], [0]))
    prefix = gl.associative_scan(present, 0, _sum_int)
    count = gl.sum(present, 0)
    ok = gl.full((), 1, gl.int32, gl.BlockedLayout([1], [32], [16], [0]))
    cached_x = load_input(X, 4096, 4, 64)
    for task in range(task_index, count * 64, ctas):
        compact = task // 64
        tile = task % 64
        expert = gl.sum(gl.where((prefix == compact + 1) & (present != 0), e, 0), 0)
        if task != SKIP:
            value = fp4_tile(
                W13, S13, cached_x, expert, tile, 8192, PAIR=True, MODE=MODE
            )
            store_projection(Inter + expert * 8 * 4096, tile, value, 4096, 64, True)
            scoped.drain_stores()
            ticket = _advance_tag(Arrivals, expert)
            if ticket % 64 == 0:
                scoped.publish_flags(Flags + expert * 128 * 32, tag, 64)
    for task in range(task_index, count * 64, ctas):
        compact = task // 64
        tile = task % 64
        expert = gl.sum(gl.where((prefix == compact + 1) & (present != 0), e, 0), 0)
        if task != SKIP:
            seen, polls = scoped.wait_flag(
                Flags + (expert * 128 + tile) * 32, tag, 100_000_000, SHADER_CLOCK=True
            )
            ok = ok & (seen == tag)
            if seen == tag:
                cached = _cached_inter(Inter, expert)
                value = fp4_tile(W2, S2, cached, expert, tile, 4096, MODE=MODE)
                store_projection(Y + expert * 8 * 4096, tile, value, 4096, 64, True)
    gl.store(Status + task_index, ok)


@g.jit
def _sum_int(a, b):
    return a + b


@g.jit
def _expert_standalone(
    X,
    Ids,
    W13,
    S13,
    W2,
    S2,
    Inter,
    Y,
    Epoch,
    Arrivals,
    Flags,
    Status,
    SKIP: gl.constexpr = -1,
    MODE: gl.constexpr = "direct",
):
    expert_stage(
        gl.program_id(0),
        gl.num_programs(0),
        X,
        Ids,
        W13,
        S13,
        W2,
        S2,
        Inter,
        Y,
        Epoch,
        Arrivals,
        Flags,
        Status,
        SKIP,
        MODE,
    )


class ExpertWorkspace(msgspec.Struct):
    inter: torch.Tensor
    output: torch.Tensor
    epochs: torch.Tensor
    arrivals: torch.Tensor
    flags: torch.Tensor
    status: torch.Tensor
    ctas: int


def make_workspace(device, *, ctas=192):
    """Allocate and zero state before capture; reuse only on one serial stream."""
    assert 128 <= ctas <= torch.cuda.get_device_properties(device).multi_processor_count
    return ExpertWorkspace(
        torch.empty((64, 8, 4096), device=device, dtype=torch.float32),
        torch.empty((64, 8, 4096), device=device, dtype=torch.float32),
        torch.zeros((ctas, 32), device=device, dtype=torch.int32),
        torch.zeros((64, 32), device=device, dtype=torch.int32),
        torch.zeros((64, 128, 32), device=device, dtype=torch.int32),
        torch.empty(ctas, device=device, dtype=torch.int32),
        ctas,
    )


def experts(x, ids, w13, s13, w2, s2, workspace, *, mode="direct", skip=-1):
    """Run local-id expert math; check every workspace status before reading output."""
    assert x.shape == (8, 4096) and x.dtype == torch.bfloat16
    assert ids.shape == (8, 8) and ids.dtype == torch.int32
    assert w13.shape == (64, 8192, 2048) and s13.shape == (64, 8192, 128)
    assert w2.shape == (64, 4096, 2048) and s2.shape == (64, 4096, 128)
    assert all(
        t.is_contiguous() and t.device == x.device for t in (x, ids, w13, s13, w2, s2)
    )
    assert all(t.dtype == torch.uint8 for t in (w13, s13, w2, s2))
    assert mode in ("alu", "unit", "direct")
    return _expert_standalone[(workspace.ctas,)](
        x,
        ids,
        w13,
        s13,
        w2,
        s2,
        workspace.inter,
        workspace.output,
        workspace.epochs,
        workspace.arrivals,
        workspace.flags,
        workspace.status,
        SKIP=skip,
        MODE=mode,
        num_warps=16,
        enable_fp_fusion=False,
    )

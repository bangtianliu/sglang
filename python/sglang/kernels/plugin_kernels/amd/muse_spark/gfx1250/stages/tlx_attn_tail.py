"""S3 tl/uTLX port: fixed norm/gate, SK4 projection, scoped virtual exchange."""

import msgspec
import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

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
    ROW1,
    BlockedEncoding,
    require_pin_support,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_projection,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_boundary import (
    norm_residual,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_attention import (
    WIDE_NORM,
    head_norm_gate,
)

NORM = tl.constexpr(BlockedEncoding([1, 8], [2, 16], [4, 1], [1, 0]))


class Workspace(tlx_workspace.S3Workspace):
    kernels: dict = msgspec.field(default_factory=dict)


def make_workspace(device):
    b = tlx_workspace.make_s3(device)
    b.state.status = torch.empty((512,), device=device, dtype=torch.int32)
    b.state.grid = 512
    return Workspace(*msgspec.structs.astuple(b))


@triton.jit
def normalize(X, Gate, Y, Failure, Seed=None):
    tile = tl.program_id(0)
    if tlx_guard.enabled(Failure, 3):
        # Same shape, distributed layout and expression as streamed_gemm._copy_a
        # with Gate and BK128. The following projection sees these rounded bits.

        row = tl.arange(0, 16)
        col = tile * 128 + tl.arange(0, 128)
        ptr = row[:, None] * 2048 + col[None, :]
        mask = tl.broadcast_to(row[:, None] < 8, (16, 128))
        x = tlx.require_layout(tl.load(X + ptr, mask, 0).to(tl.float32), NORM)
        gate = tlx.require_layout(tl.load(Gate + ptr, mask, 0).to(tl.float32), NORM)
        xx = x.reshape((16, 2, 64))
        inv = tl.rsqrt(tl.sum(xx * xx, 2) / 64 + 1e-5)
        value = (xx * inv[:, :, None]).reshape((16, 128)) / (1.0 + tl.exp(-gate))
        tlx_guard.delay(Seed, tile, 0)
        tl.store(Y + ptr, value.to(tl.bfloat16), mask)


@triton.jit
def normalize_rows(X, Gate, Y, Failure, Seed=None):
    task = tl.program_id(0)
    if tlx_guard.enabled(Failure, 3):
        row = task // 32
        head = task % 32
        d = tlx.require_layout(tl.arange(0, 64), WIDE_NORM)
        index = row * 2048 + head * 64 + d
        x = tl.load(X + index).to(tl.float32)
        g = tl.load(Gate + index).to(tl.float32)
        value = head_norm_gate(x, g)
        tlx_guard.delay(Seed, task, 0)
        tl.store(Y + index, value.to(tl.bfloat16))


# host AR: parts per row published by sibling CTAs (grid 8 * parts; 0/1: no split)
HOST_SPLIT = 4

@triton.jit
def s3_boundary(
    Partial,
    Stream,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    Status,
    Epochs,
    Scratch,
    Peers,
    Reduced,
    Failure,
    GRID: tl.constexpr,
    RANK: tl.constexpr,
    PAYLOAD: tl.constexpr,
    FLAGS: tl.constexpr,
    Seed=None,
    PIN_LAYOUT: tl.constexpr = True,
    CARRIER_RULE: tl.constexpr = False,
    STORE_REDUCED: tl.constexpr = True,
    NEXT_FLAGS=None,
    NEXT_COUNTER=None,
    NEXT_EPOCH=None,
    SPLIT: tl.constexpr = 0,
    XStamps=None,
):
    """SPLIT = parts per row (host AR only, grid 8 * SPLIT): CTA 8p + r publishes
    part p of row r (own epoch counter in word p * 32 // SPLIT of the row's slot)
    and exits."""
    pid = tl.program_id(0)
    row = pid % 8
    if SPLIT and pid >= 8:
        part = pid // 8
        tag = tlx_standin.advance(Epochs + part * (32 // SPLIT), row)
        i = tlx.require_layout(tl.arange(0, 512), ROW1)
        ok = tl.min(tl.load(Status + i), 0) == 1
        if ok:
            tlx_standin.publish_part(Partial, Peers, tag, row, part, SPLIT, GRID, RANK)
    else:
        s3_boundary_row(Partial, Stream, Pre, Alpha, Beta, Out, Updated, Status, Epochs,
                        Scratch, Peers, Reduced, Failure, GRID, RANK, PAYLOAD, FLAGS, Seed,
                        PIN_LAYOUT, CARRIER_RULE, STORE_REDUCED, NEXT_FLAGS, NEXT_COUNTER,
                        NEXT_EPOCH, SPLIT, row, XStamps)


@triton.jit
def s3_boundary_row(Partial, Stream, Pre, Alpha, Beta, Out, Updated, Status, Epochs, Scratch,
                    Peers, Reduced, Failure, GRID: tl.constexpr, RANK: tl.constexpr,
                    PAYLOAD: tl.constexpr, FLAGS: tl.constexpr, Seed, PIN_LAYOUT: tl.constexpr,
                    CARRIER_RULE: tl.constexpr, STORE_REDUCED: tl.constexpr, NEXT_FLAGS,
                    NEXT_COUNTER, NEXT_EPOCH, SPLIT: tl.constexpr, row, XStamps=None):
    XT: tl.constexpr = XStamps is not None
    e0 = tlx_standin.clock() if XT else 0
    tag = tlx_standin.advance(Epochs, row)
    i = tlx.require_layout(tl.arange(0, 512), ROW1)
    ok = tl.min(tl.load(Status + i), 0) == 1
    if ok:
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
            KEEP_CARRIER=CARRIER_RULE,
            # Value is consumed directly; the debug exchange tensor is unread.
            STORE_REDUCED=STORE_REDUCED,
            SPLIT=SPLIT,
            XStamps=XStamps,
        )
        tlx_guard.delay(Seed, row, 2)
        if ok:
            norm_residual(
                row,
                Reduced + RANK * 8 * 8192,
                Stream,
                None,
                Pre,
                Alpha,
                Beta,
                Out,
                Updated,
                False,
                False,
                Value=value,
                PIN_LAYOUT=PIN_LAYOUT,
                CARRIER_RULE=CARRIER_RULE,
                OUTPUT_AGENT=NEXT_FLAGS is not None,
            )
            if NEXT_FLAGS is not None:
                tlx_memory.drain_stores()
                arrived = tl.atomic_add(NEXT_COUNTER, 1, sem="relaxed", scope="gpu")
                if arrived == 7:
                    consumers = tl.arange(0, 512)
                    next_tag = tl.load(NEXT_EPOCH)
                    tlx_memory.store_words(
                        NEXT_FLAGS + consumers * 32,
                        tl.full((512,), next_tag, tl.int32),
                        consumers < 288,
                    )
                    tlx_memory.drain_stores()
    tl.store(Status + row, ok)
    tlx_guard.finish(Failure, ok, row, 3)
    if XT:
        xk = tl.arange(0, 2)
        tl.store(XStamps + row * 8 + 5 + xk,
                 tl.where(xk == 0, e0, tlx_standin.clock()).to(tl.int64))


# probe: int64 [64] buffer, per row: exchange entry / publish done / flags seen / payload read
# (realtime), poll count, kernel-row entry and exit (mk/layer4_xtrace.py)
S3_XTRACE = None


def attention_tail(
    x,
    gate,
    weight,
    stream,
    pre,
    alpha,
    beta,
    *,
    rank,
    workspace,
    skip=-1,
    delay_seed=None,
    output=None,
    trace=None,
    failure=None,
    guard_mode=0,
    pin_layout=True,
    store_reduced=True,
    split_norm=False,
    prenormalized=False,
):
    assert guard_mode == 0 and trace is None
    if pin_layout:
        require_pin_support()
    ws = workspace
    s = ws.state
    p = ws.peers
    out, updated = (
        (torch.empty_like(stream, dtype=torch.bfloat16), torch.empty_like(stream))
        if output is None
        else output
    )
    normalize_fn = normalize_rows if split_norm else normalize
    norm_grid = 256 if split_norm else 16
    na = (x, gate, ws.normalized, failure)
    no = dict(Seed=delay_seed, num_warps=4, enable_fp_fusion=False)
    pa = (ws.normalized, weight, ws.projection, s.status)
    po = dict(
        Failure=failure,
        N=8192,
        K=2048,
        STAGE=3,
        SKIP=skip,
        Seed=delay_seed,
        Packed=ws.partial[rank],
        num_warps=2,
        enable_fp_fusion=False,
    )
    fa = (
        ws.partial,
        stream,
        pre,
        alpha,
        beta,
        out,
        updated,
        s.status,
        s.epochs,
        p.scratch,
        p.peers,
        p.reduced,
        failure,
    )
    fo = dict(
        PIN_LAYOUT=pin_layout,
        STORE_REDUCED=store_reduced,
        GRID=p.ctas,
        RANK=rank,
        PAYLOAD=p.payload,
        FLAGS=p.flags,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
    )
    key = (
        rank,
        skip,
        delay_seed is not None,
        failure is not None,
        pin_layout,
        store_reduced,
        split_norm,
    )
    if key not in s.admitted:
        records = []
        kernels = []
        for name, fn, aa, kk, grid in [
            ("norm_gate", normalize_fn, na, no, norm_grid),
            ("projection", tlx_projection.project, pa, po, 512),
            ("AR_boundary", s3_boundary, fa, fo, 8),
        ]:
            k = fn.warmup(*aa, **kk, grid=(grid,))
            kernels.append(k)
            r = kernel_occupancy("independent-S3-" + name, k)
            assert r.active_ctas_per_cu >= 1 and k.metadata.shared <= 163840
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
        s.admitted[key] = records
        ws.kernels[key] = kernels
    if not prenormalized:  # else S2's merge already wrote ws.normalized
        normalize_fn[(norm_grid,)](*na, **no)
    tlx_projection.project[(8192 // tlx_projection.BN_S3,)](*pa, **{**po, **({"num_warps": 4} if tlx_projection.BN_S3 == 64 else {})}, EARLY=tlx_projection.EARLY_ISSUE, COPY=tlx_projection.COPY_MODE, BN=tlx_projection.BN_S3,
                                   DEPTH=tlx_projection.DEPTH_S3)
    split = int(HOST_SPLIT) if tlx_standin.HOST.value else 0
    s3_boundary[(8 * max(split, 1),)](*fa, **fo, SPLIT=split,
                                      **({} if S3_XTRACE is None else {"XStamps": S3_XTRACE}))
    return out, updated

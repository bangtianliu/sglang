"""S3 trial: exact norm/gate, two-wave TDM projection, separate AR/boundary."""

import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    chain_guard,
    standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attn_tail,
    wide_projection,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
    norm_residual,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)

_make_common_workspace = common_attn_tail.make_workspace


@g.jit
def normalize(X, Gate, Y, Failure, Seed=None):
    tile = gl.program_id(0)
    if chain_guard.enabled(Failure, 3):
        # Same shape, distributed layout and expression as streamed_gemm._copy_a
        # with Gate and BK128. The following projection sees these rounded bits.
        al: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
        row = gl.arange(0, 16, layout=gl.SliceLayout(1, al))
        col = tile * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, al))
        ptr = row[:, None] * 2048 + col[None, :]
        mask = (row[:, None] < 8).broadcast_to((16, 128))
        x = gl.load(X + ptr, mask, 0).to(gl.float32)
        gate = gl.load(Gate + ptr, mask, 0).to(gl.float32)
        xx = x.reshape((16, 2, 64))
        inv = gl.rsqrt(gl.sum(xx * xx, 2) / 64 + 1e-5)
        value = (xx * inv[:, :, None]).reshape((16, 128)) / (1.0 + gl.exp(-gate))
        delayed_point(Seed, tile, 0)
        gl.store(Y + ptr, value.to(gl.bfloat16), mask)


@g.jit
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
    GRID: gl.constexpr,
    RANK: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    Seed=None,
):
    row = gl.program_id(0)
    tag = _advance_tag(Epochs, row)
    i = gl.arange(0, 512, layout=gl.BlockedLayout([1], [32], [4], [0]))
    ok = gl.min(gl.load(Status + i), 0) == 1
    if ok:
        value, ok = standin.reduce_row(
            Partial, Scratch, Peers, Reduced, tag, row, 8192, GRID, RANK, PAYLOAD, FLAGS
        )
        delayed_point(Seed, row, 2)
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
                True,
                Value=value,
            )
    gl.store(Status + row, ok)
    chain_guard.finish_value(Failure, ok, row, 3)


def make_workspace(device):
    ws = _make_common_workspace(device)
    ws.state.status = torch.empty((512,), device=device, dtype=torch.int32)
    ws.state.grid = 512
    return ws


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
):
    assert guard_mode == 0 and trace is None
    ws = workspace
    s = ws.state
    p = ws.peers
    out, updated = (
        (torch.empty_like(stream, dtype=torch.bfloat16), torch.empty_like(stream))
        if output is None
        else output
    )
    na = (x, gate, ws.normalized, failure)
    no = dict(Seed=delay_seed, num_warps=4, enable_fp_fusion=False)
    pa = (ws.normalized, weight, ws.projection, s.status, failure)
    po = dict(
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
        GRID=p.ctas,
        RANK=rank,
        PAYLOAD=p.payload,
        FLAGS=p.flags,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
    )
    key = (rank, skip, delay_seed is not None, failure is not None)
    if key not in s.admitted:
        records = []
        for name, fn, aa, kk, grid in [
            ("norm_gate", normalize, na, no, 16),
            ("projection", wide_projection.project, pa, po, 512),
            ("AR_boundary", s3_boundary, fa, fo, 8),
        ]:
            k = fn.warmup(*aa, **kk, grid=(grid,))
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
    normalize[(16,)](*na, **no)
    wide_projection.project[(512,)](*pa, **po)
    s3_boundary[(8,)](*fa, **fo)
    return out, updated

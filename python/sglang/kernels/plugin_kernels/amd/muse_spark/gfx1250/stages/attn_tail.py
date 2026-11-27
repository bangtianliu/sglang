"""S3 four virtual TP ranks on one GPU, shared GEMM core and one-shot sum.

A128-CTA launch assigns32 CTAs per rank. Input norm/gate is cached once per
CTA; each rank owns a different o_proj shard. Serial-stream workspace only;
check all128 status words and discard state on timeout. Peers are virtual
regions in a single device allocation, never physical peer mappings.
"""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective_fast import (
    oneshot,
    oneshot_plan,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    StageState,
    delayed_point,
    handoff,
    make_state,
    norm_residual,
    validate_state,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import (
    skinny_tile,
)


class S3Workspace(msgspec.Struct):
    partial: torch.Tensor
    reduced: torch.Tensor
    state: StageState
    scratch: torch.Tensor
    peers: torch.Tensor
    ready: torch.Tensor
    payload: tuple[int, int, int]
    flags: tuple[int, int, int]


def make_workspace(device):
    plan = oneshot_plan(8192, 32, 4)
    scratch = torch.zeros(plan.total_words, device=device, dtype=torch.int32)
    regions = [plan.region(name) for name in ("payload", "flags")]
    return S3Workspace(
        torch.empty((4, 8, 8192), device=device, dtype=torch.bfloat16),
        torch.empty((4, 8, 8192), device=device, dtype=torch.float32),
        make_state(device, ctas=128, groups=4, phases=2),
        scratch,
        torch.full((4,), scratch.data_ptr(), device=device, dtype=torch.int64),
        torch.empty((1000, 128, 32), device=device, dtype=torch.int32),
        *((r.base, r.rank_stride, r.parity_stride) for r in regions),
    )


@g.jit
def _norm_gate_input(X, Gate):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 4], [1, 0])
    s = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    j = gl.arange(0, 4096, layout=gl.SliceLayout(0, layout))
    offset = (j[None, :] // 512) * 2048 + s[:, None] * 512 + j[None, :] % 512
    x = gl.reshape(gl.load(X + offset).to(gl.float32), (4, 8, 8, 64))
    gate = gl.load(Gate + offset).to(gl.float32)
    x *= gl.rsqrt(gl.sum(x * x, 3) / 64 + 1e-5)[:, :, :, None]
    value = (gl.reshape(x, (4, 4096)) * (1.0 / (1.0 + gl.exp(-gate)))).to(gl.bfloat16)
    shared = gl.allocate_shared_memory(
        gl.bfloat16, [4, 4096], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    shared.store(value)
    return shared


@g.jit
def s3_stage(
    task_index,
    ctas,
    X,
    Gate,
    W,
    Stream,
    Pre,
    Alpha,
    Beta,
    Partial,
    Reduced,
    Epochs,
    Arrivals,
    Flags,
    Status,
    Scratch,
    Peers,
    Ready,
    Out,
    NewStream,
    PAYLOAD: gl.constexpr,
    EXCHANGE_FLAGS: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
):
    gl.static_assert(ctas == 128)
    rank, local = task_index // 32, task_index % 32
    if task_index != SKIP:
        tag = _advance_tag(Epochs, task_index)
        delayed_point(Seed, task_index, 0)
        cached = _norm_gate_input(X + rank * 16384, Gate + rank * 16384)
        for tile in range(local, 128, 32):
            value = skinny_tile(
                W + rank * 8192 * 2048,
                cached,
                tile,
                8192,
                2048,
                64,
                128,
                4,
                8,
                2,
                ".cs",
                False,
            )
            layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [16, 1], [1, 0])
            r = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
            packed_layout: gl.constexpr = gl.BlockedLayout(
                [1, 4], [1, 32], [16, 1], [1, 0]
            )
            n = gl.arange(0, 32, layout=gl.SliceLayout(0, packed_layout))
            rounded = gl.convert_layout(value, layout).to(gl.bfloat16)
            low, high = gl.split(gl.reshape(rounded, (16, 32, 2)))
            packed = low.to(gl.uint16, bitcast=True).to(gl.uint32) | (
                high.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
            )
            packed = gl.convert_layout(packed, packed_layout)
            rr = gl.arange(0, 16, layout=gl.SliceLayout(1, packed_layout))
            scoped.store_words(
                Partial.to(gl.pointer_type(gl.int32))
                + rank * 32768
                + rr[:, None] * 4096
                + tile * 32
                + n[None, :],
                packed.to(gl.int32),
                rr[:, None] < 8,
            )
        delayed_point(Seed, task_index, 1)
        ready = handoff(Arrivals, Flags, tag, rank, local, 32)
        if ready:
            oneshot(
                Partial.to(gl.pointer_type(gl.int32)),
                Scratch,
                Peers,
                None,
                Reduced,
                Ready,
                None,
                8192,
                4,
                32,
                512,
                PAYLOAD,
                EXCHANGE_FLAGS,
                PROFILE=0,
                TAG=tag,
                SCOPED_INPUT=True,
                SCOPED_OUTPUT=True,
                READY_STRIDE=32,
                TASK_INDEX=task_index,
            )
            scoped.drain_stores()
            # Every rank's full reduction must be visible before a row norm.
            delayed_point(Seed, task_index, 2)
            joined = handoff(Arrivals + 4 * 32, Flags + 128 * 32, tag, rank, local, 32)
            exchange_ready = (
                scoped.load_flag(Ready + (((tag - 1) % 1000) * 128 + task_index) * 32)
                == 1
            )
            ready = ready & joined & exchange_ready
            if ready and local < 8:
                norm_residual(
                    local,
                    Reduced + rank * 65536,
                    Stream,
                    None,
                    Pre,
                    Alpha,
                    Beta,
                    Out + rank * 65536,
                    NewStream + rank * 65536,
                    False,
                    False,
                    False,
                )
        gl.store(Status + task_index, ready.to(gl.int32))


@g.jit
def _standalone(
    X,
    Gate,
    W,
    Stream,
    Pre,
    Alpha,
    Beta,
    Partial,
    Reduced,
    Epochs,
    Arrivals,
    Flags,
    Status,
    Scratch,
    Peers,
    Ready,
    Out,
    NewStream,
    PAYLOAD: gl.constexpr,
    EXCHANGE_FLAGS: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    ROTATE: gl.constexpr = 0,
):
    s3_stage(
        (gl.program_id(0) + ROTATE) % 128,
        128,
        X,
        Gate,
        W,
        Stream,
        Pre,
        Alpha,
        Beta,
        Partial,
        Reduced,
        Epochs,
        Arrivals,
        Flags,
        Status,
        Scratch,
        Peers,
        Ready,
        Out,
        NewStream,
        PAYLOAD,
        EXCHANGE_FLAGS,
        SKIP,
        Seed,
    )


def attention_tail(
    x, gate, weight, stream, pre, alpha, beta, *, workspace=None, delay_seed=None
):
    assert x.shape == gate.shape == (4, 8, 2048) and weight.shape == (4, 8192, 2048)
    assert stream.shape == (8, 8192) and stream.dtype == torch.float32
    assert all(
        t.dtype == torch.bfloat16 and t.is_contiguous() for t in (x, gate, weight)
    )
    assert pre.shape == alpha.shape == beta.shape == (8192,)
    assert pre.dtype == torch.bfloat16 and alpha.dtype == beta.dtype == torch.float32
    assert all(
        t.device == x.device and t.is_contiguous()
        for t in (gate, weight, stream, pre, alpha, beta)
    )
    if workspace is None:
        workspace = make_workspace(x.device)
    validate_state(workspace.state, x.device, ctas=128, groups=4, phases=2)
    for tensor, shape, dtype in (
        (workspace.partial, (4, 8, 8192), torch.bfloat16),
        (workspace.reduced, (4, 8, 8192), torch.float32),
        (workspace.ready, (1000, 128, 32), torch.int32),
        (workspace.peers, (4,), torch.int64),
    ):
        assert tensor.device == x.device and tensor.shape == shape
        assert tensor.dtype == dtype and tensor.is_contiguous()
    assert (
        workspace.scratch.device == x.device
        and workspace.scratch.dtype == torch.int32
        and workspace.scratch.is_contiguous()
    )
    out = torch.empty((4, 8, 8192), device=x.device, dtype=torch.bfloat16)
    updated = torch.empty((4, 8, 8192), device=x.device, dtype=torch.float32)
    s = workspace.state
    _standalone[(128,)](
        x,
        gate,
        weight,
        stream,
        pre,
        alpha,
        beta,
        workspace.partial,
        workspace.reduced,
        s.epochs,
        s.arrivals,
        s.flags,
        s.status,
        workspace.scratch,
        workspace.peers,
        workspace.ready,
        out,
        updated,
        workspace.payload,
        workspace.flags,
        Seed=delay_seed,
        num_warps=16,
        enable_fp_fusion=False,
    )
    return out, updated

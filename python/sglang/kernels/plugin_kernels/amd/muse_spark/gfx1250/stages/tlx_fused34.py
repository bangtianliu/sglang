"""MK.1: S3 boundary || S4 projection on concurrent graph branches (experimental).

The S4 projection CTAs stream their router weights into LDS *before* the S3
boundary has produced x4, then wait on a cheap in-kernel handoff:
  producer: x4 stored write-through (.wt), s_wait_storecnt, then one relaxed
            atomic add on each of KC replicated counters (no cache writeback);
  consumer: one-lane scoped poll of counter[tile % KC] >= 8 * epoch, then
            first-touch cached reads of x4.
x4 must not be read by any CTA before publication within the launch.
Math/order of both stages is unchanged (bit-exact vs the T11 chain).
"""

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard,
    tlx_memory,
    tlx_standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
    WMMA,
    WMMA_A,
    WMMA_B,
    BlockedEncoding,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_attn_tail,
    tlx_projection,
    tlx_route,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_boundary import (
    norm_residual,
)

KC = 32
LANE2 = tl.constexpr(BlockedEncoding([1], [32], [2], [0]))
A_LOAD = tl.constexpr(BlockedEncoding([1, 8], [2, 16], [2, 1], [1, 0]))


@triton.jit
def _flag2(ptr):
    """Two-wave CTA: lane 0 loads one SCOPE_DEV word; broadcast by reduction."""
    lanes = tlx.require_layout(tl.arange(0, 64), LANE2)
    saved, value = tl.inline_asm_elementwise(
        "v_mov_b32 $1, 0\nv_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\nglobal_load_b32 $1, $2, off scope:SCOPE_DEV\ns_wait_loadcnt 0\ns_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,=&v,v,v,~{scc}",
        args=[
            tlx.release_layout(ptr + tl.full(lanes.shape, 0, lanes.dtype)),
            tlx.release_layout((lanes == 0).to(tl.int32)),
        ],
        dtype=(tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    return tl.sum(value, 0)


@triton.jit
def boundary_signal(
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
    Counters,
    GRID: tl.constexpr,
    RANK: tl.constexpr,
    PAYLOAD: tl.constexpr,
    FLAGS: tl.constexpr,
    STORE_REDUCED: tl.constexpr = True,
):
    """tlx_attn_tail.s3_boundary with a write-through x4 and a counter signal."""
    row = tl.program_id(0)
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
            KEEP_CARRIER=False,
            STORE_REDUCED=STORE_REDUCED,
        )
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
                PIN_LAYOUT=True,
                CARRIER_RULE=False,
                OUTPUT_WT=True,
            )
    # Publish even on failure so consumers never spin to their bound; the
    # status words carry the failure.
    tl.store(Status + row, ok, cache_modifier=".wt")
    tlx_memory.drain_stores()
    k = tlx.require_layout(tl.arange(0, 128), ROW1)
    tl.atomic_add(Counters + k * 32, 1, mask=k < 32, sem="relaxed", scope="gpu")
    tlx_guard.finish(Failure, ok, row, 3)


@triton.jit
def project_wait(
    X,
    W,
    P,
    Status,
    Counters,
    Epoch,
    BoundaryStatus,
    Failure=None,
    N: tl.constexpr = 4608,
    K: tl.constexpr = 8192,
    STAGE: tl.constexpr = 4,
    NB: tl.constexpr = 8,
    BOUND: tl.constexpr = 400000,
    PREFETCH: tl.constexpr = True,
):
    """tlx_projection.project(SK=8, tdm, ROUND_OUTPUT=False) with W prefetched
    into NB LDS banks before the x4 handoff; A read from x4 after the wait."""
    STEPS: tl.constexpr = K // 512
    tile = tl.program_id(0)
    epoch = tl.load(Epoch + tile) + 1
    tl.store(Epoch + tile, epoch)
    allowed = tlx_guard.enabled(Failure, STAGE)
    sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [16, 512], [1, 0]
    )
    sb = tlx.local_alloc((16, 512), tl.bfloat16, NB, layout=sh)
    sa = tlx.local_alloc((16, 512), tl.bfloat16, 1, layout=sh)
    bd = tlx.make_tensor_descriptor(None, W + tile * 16 * K, [16, K], [K, 1], [16, 512])
    # 1) weights first: independent of the producer.
    if PREFETCH:
        for step in tl.static_range(NB):
            tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
    # 2) cheap handoff wait.
    target = 8 * epoch
    c = _flag2(Counters + (tile % 32) * 32)
    n = 0
    while (c < target) & (n < BOUND):
        c = _flag2(Counters + (tile % 32) * 32)
        n += 1
    ready = c >= target
    j = tl.arange(0, 512)
    ok = (
        allowed
        & ready
        & (tl.min(tl.load(BoundaryStatus + j, j < 8, 1), 0) == 1)
    )
    dep = tl.minimum(c - target, 0)  # 0; orders x4 reads after the poll
    if not PREFETCH:
        for step in tl.static_range(NB):
            tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
    r = tl.arange(0, 16)
    acc = tlx.require_layout(tl.zeros((16, 16), tl.float32), WMMA)
    parts = ()
    for step in tl.static_range(STEPS):
        k = step * 512 + tl.arange(0, 512)
        ap = tlx.require_layout(X + dep + r[:, None] * K + k[None, :], A_LOAD)
        ap = tl.max_contiguous(tl.multiple_of(ap, (1, 8)), (1, 8))
        a = tl.load(
            ap, tlx.require_layout(tl.broadcast_to(r[:, None] < 8, (16, 512)), A_LOAD), 0.0
        )
        tlx.local_store(tlx.local_view(sa, 0), a)
        tlx.tdm_async_wait(min(NB - 1, STEPS - step - 1))
        tlx.workgroup_barrier()
        aa = tlx.require_layout(tlx.local_load(tlx.local_view(sa, 0)), WMMA_A)
        bb = tlx.require_layout(
            tlx.local_load(tlx.local_trans(tlx.local_view(sb, step % NB))), WMMA_B
        )
        if step % (STEPS // 8) == 0:
            acc = tlx.require_layout(tl.zeros((16, 16), tl.float32), WMMA)
        acc = tl.dot(aa, bb, acc)
        tlx.workgroup_barrier()
        if step + NB < STEPS:
            tlx.tdm_async_load(
                bd, tlx.local_view(sb, step % NB), [0, (step + NB) * 512]
            )
        if (step + 1) % (STEPS // 8) == 0:
            parts += (acc,)
    tlx.tdm_async_wait(0)
    value = ((parts[0] + parts[4]) + (parts[2] + parts[6])) + (
        (parts[1] + parts[5]) + (parts[3] + parts[7])
    )
    nn = tile * 16 + tl.arange(0, 16)
    if ok:
        tl.store(
            P + r[:, None] * N + nn[None, :],
            value.to(tl.int32, bitcast=True),
            r[:, None] < 8,
        )
    tl.store(Status + tile, ok)
    if allowed:
        tlx_guard.finish(Failure, ok, tile, STAGE)


class State:
    def __init__(self, device):
        self.counters = torch.zeros(KC * 32, device=device, dtype=torch.int32)
        self.epoch = torch.zeros(288, device=device, dtype=torch.int32)
        self.side = torch.cuda.Stream(device=device)


def tail_route(
    x,
    gate,
    o_proj,
    stream,
    pre,
    alpha,
    beta,
    router,
    bias,
    route_norm,
    *,
    rank,
    s3,
    s4,
    st,
    output,
    failure,
    split_norm,
    store_reduced,
    nb=8,
    bound=20000,
    prefetch=True,
):
    """S3 norm+projection, then S3 boundary || S4 projection, then S4 rows/slots."""
    out, updated = output
    p, s = s3.peers, s3.state
    normalize_fn = tlx_attn_tail.normalize_rows if split_norm else tlx_attn_tail.normalize
    normalize_fn[(256 if split_norm else 16,)](
        x, gate, s3.normalized, failure, num_warps=4, enable_fp_fusion=False
    )
    tlx_projection.project[(512,)](
        s3.normalized,
        o_proj,
        s3.projection,
        s.status,
        Failure=failure,
        N=8192,
        K=2048,
        STAGE=3,
        Packed=s3.partial[rank],
        num_warps=2,
        enable_fp_fusion=False,
    )
    main = torch.cuda.current_stream()
    st.side.wait_stream(main)
    boundary_signal[(8,)](
        s3.partial,
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
        st.counters,
        GRID=p.ctas,
        RANK=rank,
        PAYLOAD=p.payload,
        FLAGS=p.flags,
        STORE_REDUCED=store_reduced,
        num_warps=4,
        enable_fp_fusion=False,
    )
    with torch.cuda.stream(st.side):
        project_wait[(288,)](
            out,
            router,
            s4.projection,
            s4.producer,
            st.counters,
            st.epoch,
            s.status,
            Failure=failure,
            NB=nb,
            BOUND=bound,
            PREFETCH=prefetch,
            num_warps=2,
            enable_fp_fusion=False,
        )
    main.wait_stream(st.side)
    ra = (s4.projection, bias, route_norm, s4.scores, s4.ids, s4.pooled, s4.producer, s4.rows_ready, failure)
    tlx_route.finish_rows[(8,)](*ra, num_warps=4, enable_fp_fusion=False)
    tlx_route.slots[(64,)](
        s4.ids,
        s4.sorted_ids,
        s4.expert_ids,
        s4.num_post,
        s4.rows_ready,
        s4.state.status,
        failure,
        LOCAL_START=rank * 64,
        GRID=s4.state.grid,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, updated


@triton.jit
def _sleep(SLEEP: tl.constexpr):
    lanes = tlx.require_layout(tl.arange(0, 128), ROW1)
    tl.inline_asm_elementwise(
        f"s_sleep {SLEEP}\nv_mov_b32 $0, 0", "=v,v", [lanes], dtype=tl.int32,
        is_pure=False, pack=1,
    )


class Wmma4(tlx.layout_encoding):
    """WMMA v3 parent for 4 warps (extra warp bit replicates the 16x16 tile)."""

    def __init__(self, operand=-1):
        self.operand = operand

    def __repr__(self):
        return f"WMMA3_N4_operand{self.operand}"

    def __eq__(self, other):
        return isinstance(other, Wmma4) and self.operand == other.operand

    def __hash__(self):
        return hash((Wmma4, self.operand))

    def to_ir(self, builder, *_):
        parent = builder.get_amd_wmma_layout(
            3, True, [[0, 1], [0, 2]], [], [], [16, 16, 32], 2
        )
        return (
            parent
            if self.operand < 0
            else builder.get_dot_operand_layout(self.operand, parent, 8)
        )


W4 = tl.constexpr(Wmma4())
W4_A = tl.constexpr(Wmma4(0))
W4_B = tl.constexpr(Wmma4(1))
A_LOAD4 = tl.constexpr(BlockedEncoding([1, 8], [2, 16], [4, 1], [1, 0]))


@triton.jit
def project_tile4(
    X, W, P, Status, Counters, Epoch, tile, ok_in,
    N: tl.constexpr, K: tl.constexpr, NB: tl.constexpr, BOUND: tl.constexpr,
    WAIT: tl.constexpr, SLEEP: tl.constexpr = 0,
):
    """4-warp S4 projection tile (same WMMA sequence as the 2-warp project)."""
    STEPS: tl.constexpr = K // 512
    sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [16, 512], [1, 0]
    )
    sb = tlx.local_alloc((16, 512), tl.bfloat16, NB, layout=sh)
    sa = tlx.local_alloc((16, 512), tl.bfloat16, 1, layout=sh)
    bd = tlx.make_tensor_descriptor(None, W + tile * 16 * K, [16, K], [K, 1], [16, 512])
    for step in tl.static_range(NB):
        tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
    dep = 0
    ok = ok_in
    if WAIT:
        epoch = tl.load(Epoch + tile) + 1
        tl.store(Epoch + tile, epoch)
        target = 8 * epoch
        c = tlx_memory.load_flag(Counters + (tile % 32) * 32)
        n = 0
        while (c < target) & (n < BOUND):
            if SLEEP > 0:
                _sleep(SLEEP)
            c = tlx_memory.load_flag(Counters + (tile % 32) * 32)
            n += 1
        ok = ok & (c >= target)
        dep = tl.minimum(c - target, 0)
    r = tl.arange(0, 16)
    acc = tlx.require_layout(tl.zeros((16, 16), tl.float32), W4)
    parts = ()
    for step in tl.static_range(STEPS):
        k = step * 512 + tl.arange(0, 512)
        ap = tlx.require_layout(X + dep + r[:, None] * K + k[None, :], A_LOAD4)
        ap = tl.max_contiguous(tl.multiple_of(ap, (1, 8)), (1, 8))
        a = tl.load(
            ap, tlx.require_layout(tl.broadcast_to(r[:, None] < 8, (16, 512)), A_LOAD4), 0.0
        )
        tlx.local_store(tlx.local_view(sa, 0), a)
        tlx.tdm_async_wait(min(NB - 1, STEPS - step - 1))
        tlx.workgroup_barrier()
        aa = tlx.require_layout(tlx.local_load(tlx.local_view(sa, 0)), W4_A)
        bb = tlx.require_layout(
            tlx.local_load(tlx.local_trans(tlx.local_view(sb, step % NB))), W4_B
        )
        if step % (STEPS // 8) == 0:
            acc = tlx.require_layout(tl.zeros((16, 16), tl.float32), W4)
        acc = tl.dot(aa, bb, acc)
        tlx.workgroup_barrier()
        if step + NB < STEPS:
            tlx.tdm_async_load(bd, tlx.local_view(sb, step % NB), [0, (step + NB) * 512])
        if (step + 1) % (STEPS // 8) == 0:
            parts += (acc,)
    tlx.tdm_async_wait(0)
    value = ((parts[0] + parts[4]) + (parts[2] + parts[6])) + (
        (parts[1] + parts[5]) + (parts[3] + parts[7])
    )
    nn = tile * 16 + tl.arange(0, 16)
    if ok:
        tl.store(P + r[:, None] * N + nn[None, :], value.to(tl.int32, bitcast=True), r[:, None] < 8)
    tl.store(Status + tile, ok)


@triton.jit
def project4(X, W, P, Status, Counters, Epoch,
             N: tl.constexpr = 4608, K: tl.constexpr = 8192, NB: tl.constexpr = 8):
    """Standalone 4-warp projection (no wait), for exactness probes."""
    project_tile4(X, W, P, Status, Counters, Epoch, tl.program_id(0), True,
                  N, K, NB, 1, False)


@triton.jit
def fused34(
    Partial, Stream, Pre, Alpha, Beta, Out, Updated, Status, Epochs, Scratch, Peers,
    Reduced, Failure, Counters, W, P, ProjStatus, ProjEpoch,
    GRID: tl.constexpr, RANK: tl.constexpr, PAYLOAD: tl.constexpr, FLAGS: tl.constexpr,
    STORE_REDUCED: tl.constexpr = True, NB: tl.constexpr = 8, BOUND: tl.constexpr = 20000,
    SLEEP: tl.constexpr = 0, RELEASE: tl.constexpr = "release", WAIT: tl.constexpr = True,
):
    """One launch: CTAs 0-7 = S3 boundary (signal), CTAs 8.. = S4 projection (wait)."""
    pid = tl.program_id(0)
    if pid < 8:
        row = pid
        tag = tlx_standin.advance(Epochs, row)
        i = tlx.require_layout(tl.arange(0, 512), ROW1)
        ok = tl.min(tl.load(Status + i), 0) == 1
        if ok:
            value, ok = tlx_standin.reduce_row(
                Partial, Scratch, Peers, Reduced, tag, row, GRID, RANK, PAYLOAD, FLAGS,
                KEEP_CARRIER=False, STORE_REDUCED=STORE_REDUCED,
            )
            if ok:
                norm_residual(
                    row, Reduced + RANK * 8 * 8192, Stream, None, Pre, Alpha, Beta,
                    Out, Updated, False, False, Value=value, PIN_LAYOUT=True,
                    CARRIER_RULE=False, OUTPUT_WT=True,
                )
        tl.store(Status + row, ok, cache_modifier=".wt")
        tlx_memory.drain_stores()
        k = tlx.require_layout(tl.arange(0, 128), ROW1)
        # release: keeps the x4/status stores ahead of the signal (8 CTAs only)
        tl.atomic_add(Counters + k * 32, 1, mask=k < 32, sem=RELEASE, scope="gpu")
        tlx_guard.finish(Failure, ok, row, 3)
    else:
        tile = pid - 8
        allowed = tlx_guard.enabled(Failure, 4)
        project_tile4(Out, W, P, ProjStatus, Counters, ProjEpoch, tile, allowed,
                      4608, 8192, NB, BOUND, WAIT, SLEEP)
        if allowed:
            tlx_guard.finish(Failure, tl.load(ProjStatus + tile) == 1, tile, 4)


def tail_route_one(
    x, gate, o_proj, stream, pre, alpha, beta, router, bias, route_norm,
    *, rank, s3, s4, st, output, failure, split_norm, store_reduced, nb=8, bound=20000,
    sleep=0, wait=True,
):
    """S3 norm+projection, ONE fused S3-boundary/S4-projection launch, S4 rows/slots."""
    out, updated = output
    p, s = s3.peers, s3.state
    normalize_fn = tlx_attn_tail.normalize_rows if split_norm else tlx_attn_tail.normalize
    normalize_fn[(256 if split_norm else 16,)](
        x, gate, s3.normalized, failure, num_warps=4, enable_fp_fusion=False
    )
    tlx_projection.project[(512,)](
        s3.normalized, o_proj, s3.projection, s.status, Failure=failure, N=8192, K=2048,
        STAGE=3, Packed=s3.partial[rank], num_warps=2, enable_fp_fusion=False,
    )
    fused34[(8 + 288,)](
        s3.partial, stream, pre, alpha, beta, out, updated, s.status, s.epochs,
        p.scratch, p.peers, p.reduced, failure, st.counters, router, s4.projection,
        s4.producer, st.epoch, GRID=p.ctas, RANK=rank, PAYLOAD=p.payload, FLAGS=p.flags,
        STORE_REDUCED=store_reduced, NB=nb, BOUND=bound, SLEEP=sleep, WAIT=wait, num_warps=4,
        enable_fp_fusion=False,
    )
    ra = (s4.projection, bias, route_norm, s4.scores, s4.ids, s4.pooled, s4.producer,
          s4.rows_ready, failure)
    tlx_route.finish_rows[(8,)](*ra, num_warps=4, enable_fp_fusion=False)
    tlx_route.slots[(64,)](
        s4.ids, s4.sorted_ids, s4.expert_ids, s4.num_post, s4.rows_ready,
        s4.state.status, failure, LOCAL_START=rank * 64, GRID=s4.state.grid,
        num_warps=4, enable_fp_fusion=False,
    )
    return out, updated


@triton.jit
def project4_wait(X, W, P, Status, Counters, Epoch,
                  N: tl.constexpr = 4608, K: tl.constexpr = 8192, NB: tl.constexpr = 8,
                  SLEEP: tl.constexpr = 0):
    """Standalone 4-warp projection WITH the wait path (for probes)."""
    project_tile4(X, W, P, Status, Counters, Epoch, tl.program_id(0), True,
                  N, K, NB, 20000, True, SLEEP)


@triton.jit
def project4n(X, W, P, Status, N: tl.constexpr = 4608, K: tl.constexpr = 8192,
              NB: tl.constexpr = 4):
    """4 warps, each owning a distinct 16-column tile: CTA = 16 x 64, no replication."""
    STEPS: tl.constexpr = K // 512
    blk = tl.program_id(0)
    sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [64, 512], [1, 0]
    )
    sha: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [16, 512], [1, 0]
    )
    sb = tlx.local_alloc((64, 512), tl.bfloat16, NB, layout=sh)
    sa = tlx.local_alloc((16, 512), tl.bfloat16, 1, layout=sha)
    bd = tlx.make_tensor_descriptor(None, W + blk * 64 * K, [64, K], [K, 1], [64, 512])
    for step in tl.static_range(NB):
        tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
    r = tl.arange(0, 16)
    acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
    parts = ()
    for step in tl.static_range(STEPS):
        k = step * 512 + tl.arange(0, 512)
        ap = tlx.require_layout(X + r[:, None] * K + k[None, :], A_LOAD4)
        ap = tl.max_contiguous(tl.multiple_of(ap, (1, 8)), (1, 8))
        a = tl.load(ap, tlx.require_layout(tl.broadcast_to(r[:, None] < 8, (16, 512)), A_LOAD4), 0.0)
        tlx.local_store(tlx.local_view(sa, 0), a)
        tlx.tdm_async_wait(min(NB - 1, STEPS - step - 1))
        tlx.workgroup_barrier()
        aa = tlx.require_layout(tlx.local_load(tlx.local_view(sa, 0)), W4_A)
        bb = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_view(sb, step % NB))), W4_B)
        if step % (STEPS // 8) == 0:
            acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
        acc = tl.dot(aa, bb, acc)
        tlx.workgroup_barrier()
        if step + NB < STEPS:
            tlx.tdm_async_load(bd, tlx.local_view(sb, step % NB), [0, (step + NB) * 512])
        if (step + 1) % (STEPS // 8) == 0:
            parts += (acc,)
    tlx.tdm_async_wait(0)
    value = ((parts[0] + parts[4]) + (parts[2] + parts[6])) + (
        (parts[1] + parts[5]) + (parts[3] + parts[7])
    )
    nn = blk * 64 + tl.arange(0, 64)
    tl.store(P + r[:, None] * N + nn[None, :], value.to(tl.int32, bitcast=True), r[:, None] < 8)
    tl.store(Status + blk * 4 + tl.arange(0, 4), 1)


def routing_p4n(x, weight, bias, norm, *, rank=0, workspace=None, failure=None, nb=4, **_):
    """tlx_route.routing with the projection replaced by the exact N-split 4-warp one."""
    ws = workspace
    failure = ws.local_failure if failure is None else failure
    project4n[(72,)](x, weight, ws.projection, ws.producer, NB=nb, num_warps=4,
                     enable_fp_fusion=False)
    tlx_route.finish_rows[(8,)](ws.projection, bias, norm, ws.scores, ws.ids, ws.pooled,
                                ws.producer, ws.rows_ready, failure, num_warps=4,
                                enable_fp_fusion=False)
    tlx_route.slots[(64,)](ws.ids, ws.sorted_ids, ws.expert_ids, ws.num_post, ws.rows_ready,
                           ws.state.status, failure, LOCAL_START=rank * 64,
                           GRID=ws.state.grid, num_warps=4, enable_fp_fusion=False)
    return ws.scores, ws.ids, ws.pooled, ws.sorted_ids, ws.expert_ids, ws.num_post


@triton.jit
def ntile_dot(X, W, blk, K: tl.constexpr, NB: tl.constexpr, SK: tl.constexpr):
    """Exact 4-warp N-split tile: rows 0..15 (8 valid) x 64 columns of X @ W^T, with
    tlx_projection.project's split-K tree. Returns fp32 [16, 64] in W4 layout."""
    STEPS: tl.constexpr = K // 512
    sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [64, 512], [1, 0]
    )
    sha: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [16, 512], [1, 0]
    )
    sb = tlx.local_alloc((64, 512), tl.bfloat16, NB, layout=sh)
    sa = tlx.local_alloc((16, 512), tl.bfloat16, 1, layout=sha)
    bd = tlx.make_tensor_descriptor(None, W + blk * 64 * K, [64, K], [K, 1], [64, 512])
    for step in tl.static_range(min(NB, STEPS)):
        tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
    r = tl.arange(0, 16)
    acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
    parts = ()
    for step in tl.static_range(STEPS):
        k = step * 512 + tl.arange(0, 512)
        ap = tlx.require_layout(X + r[:, None] * K + k[None, :], A_LOAD4)
        ap = tl.max_contiguous(tl.multiple_of(ap, (1, 8)), (1, 8))
        a = tl.load(ap, tlx.require_layout(tl.broadcast_to(r[:, None] < 8, (16, 512)), A_LOAD4), 0.0)
        tlx.local_store(tlx.local_view(sa, 0), a)
        tlx.tdm_async_wait(min(min(NB, STEPS) - 1, STEPS - step - 1))
        tlx.workgroup_barrier()
        aa = tlx.require_layout(tlx.local_load(tlx.local_view(sa, 0)), W4_A)
        bb = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_view(sb, step % NB))), W4_B)
        if step % (STEPS // SK) == 0:
            acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
        acc = tl.dot(aa, bb, acc)
        tlx.workgroup_barrier()
        if step + NB < STEPS:
            tlx.tdm_async_load(bd, tlx.local_view(sb, step % NB), [0, (step + NB) * 512])
        if (step + 1) % (STEPS // SK) == 0:
            parts += (acc,)
    tlx.tdm_async_wait(0)
    if SK == 8:
        value = ((parts[0] + parts[4]) + (parts[2] + parts[6])) + (
            (parts[1] + parts[5]) + (parts[3] + parts[7])
        )
    else:
        value = (parts[0] + parts[2]) + (parts[1] + parts[3])
    return value


@triton.jit
def project_n(X, W, P, Status, Packed, N: tl.constexpr, K: tl.constexpr, NB: tl.constexpr,
              SK: tl.constexpr, ROUND_OUTPUT: tl.constexpr):
    """Generic exact N-split projection (probe/drop-in for tlx_projection.project)."""
    blk = tl.program_id(0)
    value = ntile_dot(X, W, blk, K, NB, SK)
    if ROUND_OUTPUT:
        value = value.to(tl.bfloat16).to(tl.float32)
    r = tl.arange(0, 16)
    nn = blk * 64 + tl.arange(0, 64)
    tl.store(P + r[:, None] * N + nn[None, :], value.to(tl.int32, bitcast=True), r[:, None] < 8)
    if Packed is not None:
        tl.store(Packed + r[:, None] * N + nn[None, :], value.to(tl.bfloat16), r[:, None] < 8)
    tl.store(Status + blk * 4 + tl.arange(0, 4), 1)


@triton.jit
def ntile_dot_wait(X, W, blk, Cnt, target, K: tl.constexpr, NB: tl.constexpr,
                   SK: tl.constexpr, BOUND: tl.constexpr, SLEEP: tl.constexpr = 0,
                   PREFETCH: tl.constexpr = True, FAST: tl.constexpr = 0):
    """ntile_dot that prefetches its first NB weight banks, then waits until
    Cnt >= target (one-lane scoped poll) before reading X (first touch)."""
    STEPS: tl.constexpr = K // 512
    sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [64, 512], [1, 0]
    )
    sha: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [16, 512], [1, 0]
    )
    sb = tlx.local_alloc((64, 512), tl.bfloat16, NB, layout=sh)
    sa = tlx.local_alloc((16, 512), tl.bfloat16, 1, layout=sha)
    bd = tlx.make_tensor_descriptor(None, W + blk * 64 * K, [64, K], [K, 1], [64, 512])
    if PREFETCH:
        for step in tl.static_range(min(NB, STEPS)):
            tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
    c = tlx_memory.load_flag(Cnt)
    n = 0
    while (c < target) & (n < BOUND):
        if FAST > 0:
            # FAST polls at SLEEP, then long sleeps: the bound must outlast the
            # producer's own (host all-reduce) timeout, or a late peer makes this
            # consumer give up first and fail the whole stage
            if n < FAST:
                _sleep(SLEEP)
            else:
                _sleep(127)
        elif SLEEP > 0:
            _sleep(SLEEP)
        c = tlx_memory.load_flag(Cnt)
        n += 1
    ready = c >= target
    # 0 when ready (data dependency on the flag); 0 too on timeout, never a
    # negative offset (the result is discarded then)
    dep = tl.where(ready, tl.minimum(c - target, 0), 0)
    if not PREFETCH:
        for step in tl.static_range(min(NB, STEPS)):
            tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
    r = tl.arange(0, 16)
    acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
    parts = ()
    for step in tl.static_range(STEPS):
        k = step * 512 + tl.arange(0, 512)
        ap = tlx.require_layout(X + dep + r[:, None] * K + k[None, :], A_LOAD4)
        ap = tl.max_contiguous(tl.multiple_of(ap, (1, 8)), (1, 8))
        a = tl.load(ap, tlx.require_layout(tl.broadcast_to(r[:, None] < 8, (16, 512)), A_LOAD4), 0.0)
        tlx.local_store(tlx.local_view(sa, 0), a)
        tlx.tdm_async_wait(min(min(NB, STEPS) - 1, STEPS - step - 1))
        tlx.workgroup_barrier()
        aa = tlx.require_layout(tlx.local_load(tlx.local_view(sa, 0)), W4_A)
        bb = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_view(sb, step % NB))), W4_B)
        if step % (STEPS // SK) == 0:
            acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
        acc = tl.dot(aa, bb, acc)
        tlx.workgroup_barrier()
        if step + NB < STEPS:
            tlx.tdm_async_load(bd, tlx.local_view(sb, step % NB), [0, (step + NB) * 512])
        if (step + 1) % (STEPS // SK) == 0:
            parts += (acc,)
    tlx.tdm_async_wait(0)
    if SK == 8:
        value = ((parts[0] + parts[4]) + (parts[2] + parts[6])) + (
            (parts[1] + parts[5]) + (parts[3] + parts[7])
        )
    else:
        value = (parts[0] + parts[2]) + (parts[1] + parts[3])
    return value, ready


@triton.jit
def ntile_dot_half(X, W, tile, HALF: tl.constexpr, Cnt, target, NB: tl.constexpr,
                   BOUND: tl.constexpr, SLEEP: tl.constexpr, FAST: tl.constexpr = 0):
    """Half of an exact SK8 N-split tile (K = 8192): HALF 0 = K-chunks 0,2,4,6 ->
    (p0+p4)+(p2+p6); HALF 1 = chunks 1,3,5,7 -> (p1+p5)+(p3+p7). Prefetches its
    first NB weight banks, then waits until Cnt >= target before reading X."""
    K: tl.constexpr = 8192
    sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [64, 512], [1, 0]
    )
    sha: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [16, 512], [1, 0]
    )
    sb = tlx.local_alloc((64, 512), tl.bfloat16, NB, layout=sh)
    sa = tlx.local_alloc((16, 512), tl.bfloat16, 1, layout=sha)
    bd = tlx.make_tensor_descriptor(None, W + tile * 64 * K, [64, K], [K, 1], [64, 512])
    # local step j in 0..7 -> global step: chunk c = 2*(j//2) + HALF, step = 2c + j%2
    for j in tl.static_range(NB):
        tlx.tdm_async_load(bd, tlx.local_view(sb, j), [0, (2 * (2 * (j // 2) + HALF) + (j % 2)) * 512])
    c = tlx_memory.load_flag(Cnt)
    n = 0
    while (c < target) & (n < BOUND):
        if FAST > 0:
            # FAST polls at SLEEP, then long sleeps: the bound must outlast the
            # producer's own (host all-reduce) timeout, or a late peer makes this
            # consumer give up first and fail the whole stage
            if n < FAST:
                _sleep(SLEEP)
            else:
                _sleep(127)
        elif SLEEP > 0:
            _sleep(SLEEP)
        c = tlx_memory.load_flag(Cnt)
        n += 1
    ready = c >= target
    # 0 when ready (data dependency on the flag); 0 too on timeout, never a
    # negative offset (the result is discarded then)
    dep = tl.where(ready, tl.minimum(c - target, 0), 0)
    r = tl.arange(0, 16)
    acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
    parts = ()
    for j in tl.static_range(8):
        k = (2 * (2 * (j // 2) + HALF) + (j % 2)) * 512 + tl.arange(0, 512)
        ap = tlx.require_layout(X + dep + r[:, None] * K + k[None, :], A_LOAD4)
        ap = tl.max_contiguous(tl.multiple_of(ap, (1, 8)), (1, 8))
        a = tl.load(ap, tlx.require_layout(tl.broadcast_to(r[:, None] < 8, (16, 512)), A_LOAD4), 0.0)
        tlx.local_store(tlx.local_view(sa, 0), a)
        tlx.tdm_async_wait(min(NB - 1, 8 - j - 1))
        tlx.workgroup_barrier()
        aa = tlx.require_layout(tlx.local_load(tlx.local_view(sa, 0)), W4_A)
        bb = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_view(sb, j % NB))), W4_B)
        if j % 2 == 0:
            acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
        acc = tl.dot(aa, bb, acc)
        tlx.workgroup_barrier()
        if j + NB < 8:
            tlx.tdm_async_load(bd, tlx.local_view(sb, j % NB),
                               [0, (2 * (2 * ((j + NB) // 2) + HALF) + ((j + NB) % 2)) * 512])
        if j % 2 == 1:
            parts += (acc,)
    tlx.tdm_async_wait(0)
    # parts = chunk (HALF, HALF+2, HALF+4, HALF+6): (q0 + q2) + (q1 + q3)
    return (parts[0] + parts[2]) + (parts[1] + parts[3]), ready


@triton.jit
def ntile_dot_quarter(X, W, tile, Q, Cnt, target, BOUND: tl.constexpr, SLEEP: tl.constexpr,
                      FAST: tl.constexpr = 0, NO_DOT: tl.constexpr = False):
    """Quarter Q (0..3) of an exact SK8 N-split tile (K = 8192): K-chunks Q and Q+4
    -> p_Q + p_(Q+4). The four quarters combine as (s0 + s2) + (s1 + s3), which is
    the SK8 tree ((p0+p4)+(p2+p6)) + ((p1+p5)+(p3+p7)). The whole 256 KiB weight
    slice is loaded into LDS before the wait for X (Cnt >= target)."""
    K: tl.constexpr = 8192
    sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [64, 512], [1, 0]
    )
    sha: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [16, 512], [1, 0]
    )
    sb = tlx.local_alloc((64, 512), tl.bfloat16, 4, layout=sh)
    sa = tlx.local_alloc((16, 512), tl.bfloat16, 1, layout=sha)
    bd = tlx.make_tensor_descriptor(None, W + tile * 64 * K, [64, K], [K, 1], [64, 512])
    # bank j: chunk c = Q + 4 * (j // 2), step = 2 * c + j % 2
    for j in tl.static_range(4):
        tlx.tdm_async_load(bd, tlx.local_view(sb, j), [0, (2 * (Q + 4 * (j // 2)) + (j % 2)) * 512])
    tlx.tdm_async_wait(0)  # weights resident before the wait (trace: loaded time)
    tl_ = tlx_standin.clock()
    tw = tl_
    ready = tl_ < 0
    dep = 0
    r = tl.arange(0, 16)
    out = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
    # Two passes over the SAME code: pass 0 is a dry run during the wait (x4 rows
    # masked off: no memory traffic, result discarded) that warms this CU's
    # instruction cache; pass 1 waits for x4 and computes.
    for it in range(2):
        if it == 1:
            c = tlx_memory.load_flag(Cnt)
            n = 0
            while (c < target) & (n < BOUND):
                if FAST > 0:
                    if n < FAST:
                        _sleep(SLEEP)
                    else:
                        _sleep(127)
                elif SLEEP > 0:
                    _sleep(SLEEP)
                c = tlx_memory.load_flag(Cnt)
                n += 1
            ready = c >= target
            tw = tlx_standin.clock()
            dep = tl.where(ready, tl.minimum(c - target, 0), 0)
        rows = 8 * it
        acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
        parts = ()
        avals = ()
        for j in tl.static_range(4):
            k = (2 * (Q + 4 * (j // 2)) + (j % 2)) * 512 + tl.arange(0, 512)
            ap = tlx.require_layout(X + dep + r[:, None] * K + k[None, :], A_LOAD4)
            ap = tl.max_contiguous(tl.multiple_of(ap, (1, 8)), (1, 8))
            avals += (tl.load(ap, tlx.require_layout(tl.broadcast_to(r[:, None] < rows, (16, 512)), A_LOAD4), 0.0),)
        for j in tl.static_range(4):
            tlx.local_store(tlx.local_view(sa, 0), avals[j])
            tlx.workgroup_barrier()
            aa = tlx.require_layout(tlx.local_load(tlx.local_view(sa, 0)), W4_A)
            bb = tlx.require_layout(tlx.local_load(tlx.local_trans(tlx.local_view(sb, j))), W4_B)
            if j % 2 == 0:
                acc = tlx.require_layout(tl.zeros((16, 64), tl.float32), W4)
            acc = tl.dot(aa, bb, acc)
            tlx.workgroup_barrier()
            if j % 2 == 1:
                parts += (acc,)
        out = parts[0] + parts[1]
    return out, ready, tw, tl_

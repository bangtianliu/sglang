"""All-TLX TP4/EP4 layer, with GPU-local virtual-peer setup.

All six stages use tl/uTLX in every context; Gluon is a test oracle only.
"""

import msgspec
import torch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_peers
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_attention,
    tlx_attn_tail,
    tlx_experts,
    tlx_qkvg,
    tlx_route,
    tlx_tail,
)

ALL_TLX = (1, 2, 3, 4, 5, 6)


class Workspace(msgspec.Struct):
    s1: object
    s2: object
    s3: object
    s4: object
    s5: object
    s6: object
    buffers: dict
    failure: torch.Tensor
    context: int
    left: int
    stage_policy: tuple[str, str, str] = ("tlx", "tlx", "tlx")
    attention_backend: str = "tlx"
    projection_backend: str = "tlx"
    tlx_stages: tuple[int, ...] = ALL_TLX


def make_workspace(device, *, context, left, tlx_stages=None):
    assert tlx_stages is None or tuple(tlx_stages) == ALL_TLX, (
        "production chain requires all six TLX stages"
    )
    buffers = {
        name: torch.empty(shape, device=device, dtype=dtype)
        for name, shape, dtype in [
            ("attn", (8, 2048), torch.bfloat16),
            ("x4", (8, 8192), torch.bfloat16),
            ("mid_stream", (8, 8192), torch.float32),
            ("x6", (8, 4096), torch.bfloat16),
            ("out", (8, 8192), torch.bfloat16),
            ("stream", (8, 8192), torch.float32),
        ]
    }
    ws = Workspace(
        tlx_qkvg.make_workspace(device),
        tlx_attention.make_workspace(device, context=context, window_left=left),
        tlx_attn_tail.make_workspace(device),
        tlx_route.make_workspace(device),
        tlx_experts.make_workspace(device),
        tlx_tail.make_workspace(device),
        buffers,
        torch.zeros((2,), device=device, dtype=torch.int32),
        context,
        left,
    )

    # Keep only cold A/B wins; full128k and unmeasured32k retain four phases.
    ws.s5.fuse_norm_sum = context == 8192 or (context == 131072 and (left == 2047 or FUSE_NORM_FULL128))
    return ws


def prepare_peers(ws, partial3, partial5, *, rank):
    ws.failure.zero_()
    tlx_peers.prepare(ws.s3.peers, partial3, ws.s3.state.epochs, rank=rank)
    tlx_peers.prepare(ws.s5.peers, partial5, ws.s5.state.epochs, rank=rank)


def statuses(ws):
    return (
        ws.s1.status,
        ws.s2.cta_status,
        ws.s3.state.status,
        ws.s4.state.status,
        ws.s5.state.status,
        ws.s6.state.status,
    )


def resources(ws):
    return [
        list((stage.admitted if index < 2 else stage.state.admitted).values())
        for index, stage in enumerate((ws.s1, ws.s2, ws.s3, ws.s4, ws.s5, ws.s6))
    ]


# Monokernel step 1 (T14): S3's head norm/gate runs inside S2's merge CTAs
# (same (row, head) ownership, no cross-CTA handoff). Off = T11 production.
FUSE_MERGE_NORM = True
# Step 2: S1 heads run on the last-arriving projection tile of each head.
FUSE_S1_HEADS = False
# Step 3: S4 row finish and slot table in one launch (slot CTAs recompute ids).
FUSE_ROWS_SLOTS = True
# Step 4: S1 heads folded into S1 projection (v/gate) and S2 attention (q/k).
FUSE_S1S2 = True
# S5 row normalization folded into sum_reduce for 128k full too (measured per context)
FUSE_NORM_FULL128 = True
# S6 projection TDM depth override (None = per-context choice)
S6_DEPTH = None
# per (context, left), cold A/B 2026-11-19 (MK.14d); unmeasured contexts use 4
S6_DEPTHS = {(8192, -1): 3, (8192, 2047): 3, (131072, -1): 3, (131072, 2047): 3}


def launch(x, stream, w, m, ws, *, rank, skip_stage=-1, seed=None, guard_mode=0):
    assert 0 <= rank < 4
    assert guard_mode == 0
    pos, cos, loc, mapping, kc, vc, ks, vs, pages, length = m
    b = ws.buffers

    def opts(stage):
        return dict(
            failure=ws.failure,
            guard_mode=guard_mode,
            skip=0 if skip_stage == stage else -1,
            delay_seed=seed,
        )

    projection = tlx_qkvg
    q, k, v, gate = projection.qkvg(
        x,
        w["qkvg"],
        pos,
        cos,
        kc,
        vc,
        loc,
        mapping,
        1e-05,
        5.4730064863875,
        ks,
        vs,
        workspace=ws.s1,
        head_rows=8 if (ws.context == 8192 and ws.left == 2047) else 16,
        fused="vg" if FUSE_S1S2 else FUSE_S1_HEADS,
        **opts(1),
    )
    b.update(q=q, k=k, v=v, gate=gate)
    attention = tlx_attention
    fuse_norm = FUSE_MERGE_NORM
    attention.attention(
        q,
        kc,
        vc,
        pages,
        length,
        ks,
        vs,
        context=ws.context,
        window_left=ws.left,
        mapping=mapping,
        workspace=ws.s2,
        output=b["attn"],
        norm=(gate, ws.s3.normalized) if fuse_norm else None,
        heads=dict(p1=ws.s1.projection, pos=pos, cos=cos, loc=loc, k_out=k, eps=1e-05,
                   q_scale=5.4730064863875) if FUSE_S1S2 else None,
        **opts(2),
    )
    attn_tail = tlx_attn_tail
    attn_tail.attention_tail(
        b["attn"],
        gate,
        w["o_proj"],
        stream,
        w["pre3"],
        w["alpha3"],
        w["beta3"],
        rank=rank,
        workspace=ws.s3,
        # Cold A/B wins only at sliding8k; keep the debug-store path elsewhere.
        store_reduced=not (ws.context == 8192 and ws.left == 2047),
        # Wider independent head tasks win these measured contexts.
        split_norm=ws.context in (8192, 131072)
        and not (ws.context == 8192 and ws.left == 2047),
        output=(b["x4"], b["mid_stream"]),
        prenormalized=fuse_norm,
        **opts(3),
    )
    route = tlx_route
    route.routing(
        b["x4"],
        w["joint_router_pooler"],
        w["bias"],
        w["route_norm"],
        rank=rank,
        workspace=ws.s4,
        fused=FUSE_ROWS_SLOTS,
        mx=((lambda m: (m["xq"], m["xs"]) + ((m["xq2"], m["xs2"]) if tlx_experts.SPLIT13 else ()))
            (tlx_experts.mx_buffers(b["x4"].device)) if tlx_experts.MX13 else None),
        **opts(4),
    )
    experts = tlx_experts
    experts.s5(
        ws.s4.pooled,
        ws.s4.ids,
        ws.s4.scores,
        (ws.s4.sorted_ids, ws.s4.expert_ids, ws.s4.num_post),
        w,
        w["gamma5"],
        ws.s5,
        rank=rank,
        output=b["x6"],
        **opts(5),
    )
    tail = tlx_tail
    tail.unpool_boundary(
        b["x6"],
        w["unpooler"],
        b["mid_stream"],
        w["post6"],
        w["pre6"],
        w["alpha6"],
        w["beta6"],
        workspace=ws.s6,
        packed_handoff=True,
        projection_depth=S6_DEPTH or S6_DEPTHS.get((ws.context, ws.left), 4),
        output=(b["out"], b["stream"]),
        **opts(6),
    )
    return (b["out"], b["stream"])

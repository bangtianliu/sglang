"""WIP TP4/EP4 layer: independently budgeted stages, GPU-local stand-ins.

No model/distributed initialization, graph capture or peer allocation occurs here.
Caller owns weights/metadata and captures launch() after warm admission. Real peer
transport remains disabled. Failure propagation is checked at dispatch boundaries.
"""

import msgspec
import torch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
    CommonShape,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attention as s2,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attn_tail as s3,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_experts as s5,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_qkvg as s1,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_route as s4,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_tail as s6,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    independent_attn_tail,
    independent_experts,
    independent_qkvg,
    independent_tail,
    split_attention,
)


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
    stage_policy: tuple[str, str, str]
    attention_backend: str = "split"
    projection_backend: str = "common"


def make_workspace(
    device,
    *,
    context,
    left,
    shape=None,
    stage_policy=None,
    attention_backend="split",
    projection_backend=None,
):
    assert attention_backend in ("common", "split")
    projection_backend = (
        ("wide" if (context, left) == (8192, 2047) else "common")
        if projection_backend is None
        else projection_backend
    )
    assert projection_backend in ("common", "wide")
    projection = independent_qkvg if projection_backend == "wide" else s1
    attention = split_attention if attention_backend == "split" else s2
    stage_policy = (
        (
            "wide"
            if (context, left) in ((8192, 2047), (131072, -1), (131072, 2047))
            else "common",
            "k9"
            if (context, left)
            in ((8192, -1), (8192, 2047), (131072, -1), (131072, 2047))
            else "common",
            "wide",
        )
        if stage_policy is None
        else stage_policy
    )
    assert (
        stage_policy[0] in ("common", "wide")
        and stage_policy[1] in ("common", "k9")
        and stage_policy[2]
        in (
            "common",
            "wide",
        )
    )
    attn_tail = independent_attn_tail if stage_policy[0] == "wide" else s3
    experts = independent_experts if stage_policy[1] == "k9" else s5
    tail = independent_tail if stage_policy[2] == "wide" else s6
    shape = CommonShape() if shape is None else shape
    b = {
        n: torch.empty(size, device=device, dtype=dtype)
        for n, size, dtype in [
            ("attn", (8, 2048), torch.bfloat16),
            ("x4", (8, 8192), torch.bfloat16),
            ("mid_stream", (8, 8192), torch.float32),
            ("x6", (8, 4096), torch.bfloat16),
            ("out", (8, 8192), torch.bfloat16),
            ("stream", (8, 8192), torch.float32),
        ]
    }
    return Workspace(
        projection.make_workspace(device, shape=shape),
        attention.make_workspace(
            device, context=context, window_left=left, shape=shape
        ),
        attn_tail.make_workspace(
            device, **({"shape": shape} if attn_tail is s3 else {})
        ),
        s4.make_workspace(device, shape=shape),
        experts.make_workspace(device, **({"shape": shape} if experts is s5 else {})),
        tail.make_workspace(device, **({"shape": shape} if tail is s6 else {})),
        b,
        torch.zeros((2,), device=device, dtype=torch.int32),
        context,
        left,
        stage_policy,
        attention_backend,
        projection_backend,
    )


def prepare_peers(ws, partial3, partial5, *, rank):
    ws.failure.zero_()
    standin.prepare(ws.s3.peers, partial3, ws.s3.state.epochs, rank=rank)
    standin.prepare(ws.s5.peers, partial5, ws.s5.state.epochs, rank=rank)


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
        list((stage.admitted if i < 2 else stage.state.admitted).values())
        for i, stage in enumerate((ws.s1, ws.s2, ws.s3, ws.s4, ws.s5, ws.s6))
    ]


def launch(x, stream, w, m, ws, *, rank, skip_stage=-1, seed=None, guard_mode=0):
    # Guard modes 1/2 and runtime mode -1 are diagnostic controls; 0 retains
    # the accepted status-reload protocol. Mode 2 is not fail-stop.
    assert 0 <= rank < 4
    assert guard_mode in (-1, 0, 1, 2)
    assert guard_mode != -1 or ws.failure.numel() >= 3
    pos, cos, loc, mapping, kc, vc, ks, vs, pages, length = m
    b = ws.buffers

    def opts(stage):
        return dict(
            failure=ws.failure,
            guard_mode=guard_mode,
            skip=0 if skip_stage == stage else -1,
            delay_seed=seed,
        )

    projection = independent_qkvg if ws.projection_backend == "wide" else s1
    q, k, v, gate = projection.qkvg(
        x,
        w["qkvg"],
        pos,
        cos,
        kc,
        vc,
        loc,
        mapping,
        1e-5,
        5.4730064863875,
        ks,
        vs,
        workspace=ws.s1,
        **opts(1),
    )
    b.update(q=q, k=k, v=v, gate=gate)
    attention = split_attention if ws.attention_backend == "split" else s2
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
        **opts(2),
    )
    attn_tail = independent_attn_tail if ws.stage_policy[0] == "wide" else s3
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
        output=(b["x4"], b["mid_stream"]),
        **opts(3),
    )
    s4.routing(
        b["x4"],
        w["joint_router_pooler"],
        w["bias"],
        w["route_norm"],
        rank=rank,
        workspace=ws.s4,
        **opts(4),
    )
    experts = independent_experts if ws.stage_policy[1] == "k9" else s5
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
    tail = independent_tail if ws.stage_policy[2] == "wide" else s6
    tail.unpool_boundary(
        b["x6"],
        w["unpooler"],
        b["mid_stream"],
        w["post6"],
        w["pre6"],
        w["alpha6"],
        w["beta6"],
        workspace=ws.s6,
        output=(b["out"], b["stream"]),
        **opts(6),
    )
    return b["out"], b["stream"]

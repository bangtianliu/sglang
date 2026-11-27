"""Existing common-stage APIs vs runtime-masked fused body on exact archive inputs."""

import argparse
import functools
import json

import torch
from bench_isolate46 import OUTPUTS, exact, factory, table_values
from layer_fixture import require_gpu0
from mono_case45 import LOG, case
from paired45 import measure

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import standin
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


def existing(f, w, m, old, stage):
    modules = (s1, s2, s3, s4, s5, s6)
    mod = modules[stage - 1]
    ws = (
        mod.make_workspace("cuda", context=8192, window_left=-1)
        if stage == 2
        else mod.make_workspace("cuda")
    )
    b = {n: t.clone() for n, t in old["buffers"].items()}
    ids = old["ids"].clone()
    scores = old["scores"].clone()
    pool = old["pool"].clone()
    table = table_values(old["ids"])
    holder = []
    if stage == 1:
        fn = functools.partial(
            s1.qkvg,
            f.x,
            w["qkvg"],
            f.positions,
            m[1],
            f.cache_k,
            f.cache_v,
            f.locations,
            m[3],
            1e-5,
            5.4730064863875,
            f.k_scale,
            f.v_scale,
            workspace=ws,
        )
    elif stage == 2:
        fn = functools.partial(
            s2.attention,
            b["q"],
            f.cache_k,
            f.cache_v,
            f.page_table,
            f.seq_lens,
            f.k_scale,
            f.v_scale,
            context=8192,
            mapping=m[3],
            workspace=ws,
            output=b["attn"],
        )
    elif stage == 3:
        fn = functools.partial(
            s3.attention_tail,
            b["attn"],
            b["gate"],
            w["o_proj"],
            f.stream,
            w["pre3"],
            w["alpha3"],
            w["beta3"],
            rank=0,
            workspace=ws,
            output=(b["x4"], b["mid_stream"]),
        )
    elif stage == 4:
        fn = functools.partial(
            s4.routing,
            b["x4"],
            w["joint_router_pooler"],
            w["bias"],
            w["route_norm"],
            rank=0,
            workspace=ws,
        )
    elif stage == 5:
        fn = functools.partial(
            s5.s5, pool, ids, scores, table, w, w["gamma5"], ws, rank=0
        )
    else:
        fn = functools.partial(
            s6.unpool_boundary,
            b["x6"],
            w["unpooler"],
            b["mid_stream"],
            w["post6"],
            w["pre6"],
            w["alpha6"],
            w["beta6"],
            workspace=ws,
            output=(b["out"], b["stream"]),
        )

    def prepare():
        for n, t in old["buffers"].items():
            b[n].copy_(t)
        ids.copy_(old["ids"])
        scores.copy_(old["scores"])
        pool.copy_(old["pool"])
        for name in ("cache_k", "cache_v"):
            getattr(f, name).view(torch.uint8).reshape(-1, 4, 64)[f.locations] = old[
                "cache"
            ][name]
        for n in OUTPUTS[stage]:
            b[n].fill_(float("nan"))
        if stage == 1:
            ws.state.zero_()
            ws.barriers.zero_()
            ws.status.fill_(-3)
        elif stage == 2:
            ws.epochs.zero_()
            ws.counters.zero_()
            ws.status.fill_(-3)
        else:
            ws.state.epochs.zero_()
            ws.state.barriers.zero_()
            ws.state.status.fill_(-3)
            if stage in (3, 5):
                standin.prepare(
                    ws.peers, old[f"partial{stage}"], ws.state.epochs, rank=0
                )
            if stage == 4:
                ws.ids.fill_(-1)
                ws.scores.fill_(float("nan"))
                ws.pooled.fill_(float("nan"))

    def call():
        out = fn()
        if stage == 1:
            holder[:] = out

    def check():
        status = ws.status if stage in (1, 2) else ws.state.status
        exact(status, torch.ones_like(status), f"S{stage} existing status")
        if stage == 1:
            for n, t in zip(("q", "k", "v", "gate"), holder):
                exact(t, old["buffers"][n], f"S1 existing {n}")
        elif stage == 4:
            exact(ws.ids, old["ids"], "S4 existing IDs")
            exact(ws.scores, old["scores"], "S4 existing scores")
            exact(ws.pooled, old["pool"], "S4 existing pool")
        elif stage == 5:
            exact(
                ws.peers.reduced[0].bfloat16(),
                old["buffers"]["x6"],
                "S5 existing reduced BF16",
            )
        else:
            for n in OUTPUTS[stage]:
                exact(b[n], old["buffers"][n], f"S{stage} existing {n}")
        for name in ("cache_k", "cache_v"):
            exact(
                getattr(f, name).view(torch.uint8).reshape(-1, 4, 64)[f.locations],
                old["cache"][name],
                name + " existing bytes",
            )

    def resources():
        return list((ws.admitted if stage in (1, 2) else ws.state.admitted).values())

    return (prepare, call, check), resources


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=range(1, 7), required=True)
    a = p.parse_args()
    f, w, m, old = case()
    fused, meta, k = factory(f, w, m, old, 1 << (a.stage - 1))
    own, resources = existing(f, w, m, old, a.stage)
    result, _ = measure({"existing": own, "fused": fused})
    data = dict(
        stage=a.stage,
        timing=result,
        resources=dict(existing=resources(), fused=meta),
        input="exact accepted-layer archived intermediates, separate workspaces",
        note="Existing S5 returns FP32 reduced values; BF16 boundary cast checked outside timed range, as in earlier standalone API benchmark.",
    )
    (LOG / f"phase46-existing-S{a.stage}.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                stage=a.stage,
                summary={
                    k: {n: v for n, v in x.items() if n != "samples_by_order_us"}
                    for k, x in result.items()
                },
                resources=data["resources"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

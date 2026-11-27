"""Per-CTA common-stage timelines and first-poll stand-in evidence on GPU0."""

import argparse
import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from check_experts import cases
from check_experts_complete import reference, table_for
from layer_fixture import require_gpu0
from rank_timing import rank_times
from reference_archive import checked_load

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attn_tail,
    common_experts,
    common_qkvg,
    common_tail,
)

ROOT = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["S1", "S3", "S5", "S6"], required=True)
    p.add_argument("--label", default="before")
    a = p.parse_args()
    trace = torch.zeros((768, 32), device="cuda", dtype=torch.int64)
    tasks = (
        torch.zeros((8192, 32), device="cuda", dtype=torch.int64)
        if a.stage == "S5"
        else None
    )
    setup = lambda: None
    if a.stage == "S1":
        import check_s1 as suite

        suite.ARCHIVE = True
        f = suite.build_layer(seed=7, context=8192)
        ws = common_qkvg.make_workspace("cuda")
        suite.qkvg = lambda *args, **kw: common_qkvg.qkvg(
            *args, **kw, trace=trace if tracing[0] else None
        )
        call = lambda: suite.candidate(
            f, rope=False, mapping=False, ctas=768, workspace=ws
        )
        status = ws.status
    elif a.stage == "S3":
        d = checked_load("S3-inputs-7.pt")
        ref = checked_load("S3-native-7.pt")
        ws = common_attn_tail.make_workspace("cuda")
        status = ws.state.status
        setup = lambda: standin.prepare(
            ws.peers, ref[1]["partial"], ws.state.epochs, rank=0
        )
        call = lambda: common_attn_tail.attention_tail(
            d["x"][0],
            d["gate"][0],
            d["weight"][0],
            d["stream"],
            d["pre"],
            d["alpha"],
            d["beta"],
            rank=0,
            workspace=ws,
            trace=trace if tracing[0] else None,
        )
    elif a.stage == "S5":
        d = checked_load("phase3e1-inputs-7.pt")
        x = d["x"]
        weights = {
            k: torch.stack([v.roll(r * 11, 0) for r in range(4)])
            for k, v in d["weights"].items()
        }
        gen = torch.Generator().manual_seed(7)
        gamma = (torch.randn(4096, generator=gen) * 0.02).cuda()
        scores = torch.rand((8, 8), generator=gen).cuda()
        scores /= scores.sum(1, keepdim=True)
        ids = cases(7, "cuda")["typical"]
        tt = table_for(ids)
        ref = reference(x, ids, scores, weights, gamma, tt)
        ws = common_experts.make_workspace("cuda")
        status = ws.state.status
        setup = lambda: standin.prepare(
            ws.peers, ref["partial"], ws.state.epochs, rank=0
        )
        call = lambda: common_experts.s5(
            x,
            ids,
            scores,
            tuple(t.reshape(4, -1)[0] for t in tt),
            {k: v[0] for k, v in weights.items()},
            gamma,
            ws,
            rank=0,
            trace=trace if tracing[0] else None,
            task_trace=tasks if tracing[0] else None,
        )
    else:
        d = checked_load("S6-inputs-7.pt")
        ws = common_tail.make_workspace("cuda")
        status = ws.state.status
        call = lambda: common_tail.unpool_boundary(
            *(d[k] for k in ("x", "weight", "stream", "post", "pre", "alpha", "beta")),
            workspace=ws,
            trace=trace if tracing[0] else None,
        )
    tracing = [False]
    timing = rank_times(setup, call) if a.stage in ("S3", "S5") else graph_times(call)
    tracing[0] = True
    # Trace runs are separately measured: instrumentation has a visible cost.
    traced_timing = (
        rank_times(setup, call, samples=10)
        if a.stage in ("S3", "S5")
        else graph_times(call, samples=10)
    )
    trace.zero_()
    if tasks is not None:
        tasks.zero_()
    setup()
    result = call()
    torch.cuda.synchronize()
    assert status.eq(1).all()
    raw = trace.cpu()
    valid = raw[:, 0] > 0
    origin = int(raw[valid, 0].min())
    phases = {}
    for c in range(32):
        if c in (22, 23):
            continue
        vals = raw[:, c]
        v = vals[vals > 0]
        if v.numel():
            phases[c] = dict(
                ctas=v.numel(),
                min_us=float((v.min() - origin) / 100),
                median_us=float((v.double().median() - origin) / 100),
                max_us=float((v.max() - origin) / 100),
            )
    row = dict(
        stage=a.stage,
        label=a.label,
        timing=timing,
        traced_timing=traced_timing,
        phases=phases,
        raw_ticks=raw.tolist(),
    )
    if a.stage in ("S3", "S5"):
        active = raw[raw[:, 22] > 0]
        row["poll_counts"] = dict(
            consumers=len(active),
            min=int(active[:, 22].min()),
            max=int(active[:, 22].max()),
            first_peer_mask=sorted(active[:, 23].unique().tolist()),
            all_first_poll_peers_ready=bool(((active[:, 23] & 14) == 14).all()),
        )
    if tasks is not None:
        task_raw = tasks.cpu()
        task_raw = task_raw[(task_raw[:, 0] > 0) | (task_raw[:, 3] > 0)]
        task_path = ROOT / f"phase40b-S5-{a.label}-tasks.json"
        task_path.write_text(
            json.dumps(dict(grid=768, raw_ticks=task_raw.tolist())) + "\n"
        )
        row["task_trace_path"] = str(task_path)
    path = ROOT / f"phase40b-{a.stage}-{a.label}.json"
    path.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "raw_ticks"}), flush=True)


if __name__ == "__main__":
    main()

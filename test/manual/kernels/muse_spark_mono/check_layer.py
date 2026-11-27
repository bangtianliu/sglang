"""GPU0 layer composition, bounded delayed replays and missing-producer checks."""

import argparse
import json
from pathlib import Path

import torch
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import (
    build,
    compare,
    exact_check,
    require,
    routing_check,
    teacher_reference,
)
from reference_archive import checked_load

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer

ROOT = Path("/home/bangtliu/coord/logs")


def emit(row):
    print(json.dumps(row), flush=True)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    for name, default in (
        ("seed", 7),
        ("context", 8192),
        ("left", -1),
        ("rank", 0),
        ("replays", 0),
    ):
        p.add_argument("--" + name, type=int, default=default)
    for name in ("compile-only", "all-ranks", "missing", "timing"):
        p.add_argument("--" + name, action="store_true")
    a = p.parse_args()
    f, w, m = make(a.seed, a.context, a.left, a.rank)
    stem = f"phase42-layer-{a.seed}-{a.context}-{a.left}" + (
        "-missing" if a.missing else ""
    )
    records = []
    if not a.compile_only:
        base_ref = build(f, w, a.seed, a.context, a.left)
        ref = base_ref
        emit(dict(event="reference-built", experts=len(ref["expert_oracle"])))
    base_experts = {
        n: t for n, t in w.items() if n in ("w13", "w13_scale", "w2", "w2_scale")
    }
    w3 = checked_load(f"S3-inputs-{a.seed}.pt")["weight"]
    for rank in range(4) if a.all_ranks else (a.rank,):
        if not a.compile_only:
            ref = dict(base_ref)
            ref["_rank"] = rank
        w["o_proj"] = w3[rank]
        for n, t in base_experts.items():
            w[n] = t.roll((rank - a.rank) * 11, 0).contiguous()
        ws = layer.make_workspace("cuda", context=a.context, left=a.left)
        kernel = layer.launch(f.x, f.stream, w, m, ws, rank=rank, compile_only=True)
        resource = ws.shape.admit("layer", kernel, grid=ws.shape.grid("cuda"))
        emit(
            dict(
                event="compiled",
                rank=rank,
                vgprs=kernel.n_regs,
                lds=kernel.metadata.shared,
                spills=kernel.n_spills,
                occupancy=resource.active_ctas_per_cu,
            )
        )
        for fmt in ("amdgcn", "ttgir"):
            (ROOT / f"phase41-layer-{a.left}-r{rank}.{fmt}").write_text(kernel.asm[fmt])
        if a.compile_only:
            continue
        layer.prepare_peers(ws, ref["partial3"], ref["partial5"], rank=rank)
        out = layer.launch(f.x, f.stream, w, m, ws, rank=rank)
        torch.cuda.synchronize()
        state = ws.status[:6].sum(1).tolist()
        diag = ws.diagnostics[ws.diagnostics[:, 0] != 0].cpu().tolist()
        emit(
            dict(
                event="status",
                rank=rank,
                status=state,
                diagnostics=diag[:20],
                diagnostic_records=layer.diagnostic_records(ws)[:20],
            )
        )
        exact_check(ws.status[:6], 1, "stage status", ref)
        routes = routing_check(ws, ref, w)
        if any(not row["set_equal"] for row in routes):
            ref = teacher_reference(f, w, ref, ws, rank, routes)
            emit(
                dict(
                    event="teacher-forced-reference",
                    rank=rank,
                    rows=[r for r in routes if not r["set_equal"]],
                    oracle_experts=ref["teacher_oracle_experts"],
                    peer_policy="same stand-in partials; native local S5 from candidate pool/IDs/scores; only accepted cutoff rows use conditioned S5/S6 values",
                )
            )
            layer.prepare_peers(ws, ref["partial3"], ref["partial5"], rank=rank)
            layer.launch(f.x, f.stream, w, m, ws, rank=rank)
            torch.cuda.synchronize()
            exact_check(ws.status[:6], 1, "stage status", ref)
        errors = compare(f, ws, ref, w)
        record = dict(
            seed=a.seed,
            context=a.context,
            left=a.left,
            rank=rank,
            errors=errors,
            teacher_forced=ref.get("teacher_forced", False),
            teacher_rows=ref.get("teacher_rows", []),
            resources=list(ws.admitted.values()),
        )
        emit(dict(event="correctness", **record))
        records.append(record)
        if a.replays:
            delay = torch.tensor([a.seed], device="cuda", dtype=torch.int32)
            baseline = tuple(t.clone() for t in out)
            cache_baseline = (
                f.cache_k.view(torch.uint8).clone(),
                f.cache_v.view(torch.uint8).clone(),
            )
            route_baseline = ws.s4.ids.clone()
            for i in range(a.replays):
                delay.fill_(a.seed + i * 7919)
                layer.prepare_peers(ws, ref["partial3"], ref["partial5"], rank=rank)
                layer.launch(f.x, f.stream, w, m, ws, rank=rank, seed=delay)
                torch.cuda.synchronize()
                exact_check(ws.status[:6], 1, f"replay {i} stage status", ref)
                exact_check(ws.s4.ids, route_baseline, f"replay {i} routing IDs", ref)
                for name, t, b in zip(("out", "stream"), out, baseline):
                    exact_check(t, b, f"replay {i} {name}", ref)
                for name, actual, expected in zip(
                    ("cache_k", "cache_v"), (f.cache_k, f.cache_v), cache_baseline
                ):
                    exact_check(
                        actual.view(torch.uint8), expected, f"replay {i} {name}", ref
                    )
            record["replays"] = a.replays
            emit(
                dict(
                    event="delayed-replays",
                    rank=rank,
                    launches=a.replays,
                    exact=True,
                    resources=list(ws.admitted.values()),
                )
            )
        if a.missing:
            for stage in range(1, 7):
                bad = layer.make_workspace("cuda", context=a.context, left=a.left)
                bad.buffers["out"].fill_(float("nan"))
                bad.buffers["stream"].fill_(float("nan"))
                layer.prepare_peers(bad, ref["partial3"], ref["partial5"], rank=rank)
                layer.launch(f.x, f.stream, w, m, bad, rank=rank, skip_stage=stage)
                torch.cuda.synchronize()
                diag = bad.diagnostics[bad.diagnostics[:, 0] != 0].cpu().tolist()
                require(
                    bool(diag) and not bool(bad.status[:6].eq(1).all()),
                    "missing-producer diagnostics/status",
                    ref,
                    (stage - 1, 0),
                    reason="missing producer not detected",
                )
                for name in ("out", "stream"):
                    exact_check(
                        bad.buffers[name].isnan(),
                        True,
                        f"{name} missing-producer NaN sentinel",
                        ref,
                    )
                emit(
                    dict(
                        event="missing-producer",
                        rank=rank,
                        stage=stage,
                        status=bad.status[:6].sum(1).tolist(),
                        diagnostics=diag[:8],
                        diagnostic_records=layer.diagnostic_records(bad)[:8],
                        resources=list(bad.admitted.values()),
                    )
                )
                del bad
        if a.timing:
            # Prewritten stand-ins remain constant; their ready tags follow every launch.
            # Include only the layer launch between events; preparations occur before start.
            timings = {}
            for prefetch in (False, True):
                # Warm admission before capturing the two preparation kernels and
                # one layer launch. External event nodes bracket only the layer.
                layer.prepare_peers(ws, ref["partial3"], ref["partial5"], rank=rank)
                layer.launch(f.x, f.stream, w, m, ws, rank=rank, prefetch=prefetch)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    layer.prepare_peers(ws, ref["partial3"], ref["partial5"], rank=rank)
                    start.record()
                    layer.launch(f.x, f.stream, w, m, ws, rank=rank, prefetch=prefetch)
                    end.record()
                samples = []
                for i in range(110):
                    graph.replay()
                    torch.cuda.synchronize()
                    exact_check(ws.status[:6], 1, "stage status", ref)
                    if i >= 10:
                        samples.append(start.elapsed_time(end) * 1000)
                compare(f, ws, ref, w)
                samples.sort()
                timings[str(prefetch)] = dict(
                    median_us=(samples[49] + samples[50]) / 2,
                    p99_us=samples[98],
                    method="graph-external-event-nodes-layer-only",
                )
            trace = torch.zeros(
                (ws.shape.grid("cuda"), 32), device="cuda", dtype=torch.uint64
            )
            layer.prepare_peers(ws, ref["partial3"], ref["partial5"], rank=rank)
            layer.launch(f.x, f.stream, w, m, ws, rank=rank, trace=trace)
            torch.cuda.synchronize()
            exact_check(ws.status[:6], 1, "stage status", ref)
            compare(f, ws, ref, w)
            torch.save(
                dict(trace=trace.cpu(), metrics=ws.metrics.cpu()),
                ROOT / f"{stem}-r{rank}-trace.pt",
            )
            record["timing"] = timings
            emit(dict(event="timing", rank=rank, timing=timings))
        (ROOT / f"{stem}.json").write_text(json.dumps(records, indent=2) + "\n")
        del ws
    emit(dict(event="PASS", seed=a.seed, context=a.context, left=a.left))


if __name__ == "__main__":
    main()

"""Phase4.4 bitwise checks and timings against the immutable accepted layer."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import exact_check, require
from reference_archive import checked_load, gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer

ROOT = Path("/home/bangtliu/coord")
BASE = ROOT / "phase44-baseline"
LOG = ROOT / "logs"


def verify(f, ws, old, rank, digest=False):
    ref = {"_rank": rank}
    exact_check(ws.status[:6], 1, "stage status", ref)
    exact_check(ws.s4.ids, old["ids"], "routing IDs", ref)
    for name, expected in old["buffers"].items():
        actual = ws.buffers[name]
        dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
        exact_check(actual.view(dtype), expected.view(dtype), name + " bits", ref)
    exact_check(
        ws.s4.scores.view(torch.int32),
        old["scores"].view(torch.int32),
        "routing scores bits",
        ref,
    )
    exact_check(
        ws.s4.pooled.view(torch.int16), old["pool"].view(torch.int16), "pool bits", ref
    )
    slots = f.mapping[f.locations] if ws.left >= 0 else f.locations
    for name in ("cache_k", "cache_v"):
        v = getattr(f, name).view(torch.uint8)
        exact_check(
            v.reshape(-1, 4, 64)[slots], old["cache"][name], name + " new bytes", ref
        )
        if digest:
            require(
                hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()
                == old["cache_sha256"][name],
                name,
                ref,
                ("whole-cache SHA256",),
            )


def timing(f, w, m, ws, old, rank, prefetch):
    prepare = lambda: layer.prepare_peers(
        ws, old["partial3"], old["partial5"], rank=rank
    )
    call = lambda: layer.launch(f.x, f.stream, w, m, ws, rank=rank, prefetch=prefetch)
    prepare()
    call()
    torch.cuda.synchronize()
    verify(f, ws, old, rank)
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare()
        start.record()
        call()
        end.record()
    samples = []
    for i in range(110):
        graph.replay()
        torch.cuda.synchronize()
        exact_check(ws.status[:6], 1, "timing status", {"_rank": rank})
        if i >= 10:
            samples.append(start.elapsed_time(end) * 1000)
    verify(f, ws, old, rank)
    samples.sort()
    return dict(median_us=(samples[49] + samples[50]) / 2, p99_us=samples[98])


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    for n, d in (
        ("seed", 7),
        ("context", 8192),
        ("left", -1),
        ("rank", 0),
        ("replays", 0),
    ):
        p.add_argument("--" + n, type=int, default=d)
    for n in ("all-ranks", "compile-only", "timing", "missing", "profile"):
        p.add_argument("--" + n, action="store_true")
    a = p.parse_args()
    path = BASE / f"layer-{a.seed}-{a.context}-{a.left}.pt"
    manifest = json.loads((BASE / "manifest.json").read_text())
    require(
        hashlib.sha256(path.read_bytes()).hexdigest()
        == manifest["artifacts"][path.name]["sha256"],
        "baseline SHA256",
        {"_rank": a.rank},
        (path.name,),
    )
    saved = torch.load(path, weights_only=True, map_location="cpu")
    f, w, m = make(a.seed, a.context, a.left, 0)
    wo = checked_load(f"S3-inputs-{a.seed}.pt")["weight"]
    ew = {n: w[n] for n in ("w13", "w13_scale", "w2", "w2_scale")}
    records = []
    for rank in range(4) if a.all_ranks else (a.rank,):
        ref = {"_rank": rank}
        old = gpu(saved["rows"][rank])
        w["o_proj"] = wo[rank]
        for n, t in ew.items():
            w[n] = t.roll(rank * 11, 0).contiguous()
        ws = layer.make_workspace("cuda", context=a.context, left=a.left, variant=3)
        grid = ws.shape.grid("cuda")
        k = layer.launch(f.x, f.stream, w, m, ws, rank=rank, compile_only=True)
        resource = ws.shape.admit("phase44-layer", k, grid=grid)
        print(
            json.dumps(
                dict(
                    event="compiled",
                    rank=rank,
                    vgprs=k.n_regs,
                    spills=k.n_spills,
                    lds=k.metadata.shared,
                    occupancy=resource.active_ctas_per_cu,
                )
            ),
            flush=True,
        )
        for fmt in ("amdgcn", "ttgir", "llir"):
            (LOG / f"phase44-layer-{a.left}-r{rank}.{fmt}").write_text(k.asm[fmt])
        if a.compile_only:
            continue
        layer.prepare_peers(ws, old["partial3"], old["partial5"], rank=rank)
        layer.launch(f.x, f.stream, w, m, ws, rank=rank)
        torch.cuda.synchronize()
        print(
            json.dumps(
                dict(
                    event="status",
                    rank=rank,
                    status=ws.status[:6].sum(1).tolist(),
                    diagnostics=layer.diagnostic_records(ws)[:8],
                )
            ),
            flush=True,
        )
        try:
            verify(f, ws, old, rank, True)
        except AssertionError:
            torch.save(
                dict(
                    rank=rank,
                    buffers={
                        n: t.cpu()
                        for n, t in ws.buffers.items()
                        if not n.startswith("f")
                    },
                    ids=ws.s4.ids.cpu(),
                    scores=ws.s4.scores.cpu(),
                    pool=ws.s4.pooled.cpu(),
                    row3=ws.row3.cpu(),
                    row4=ws.row4.cpu(),
                    row6=ws.row6.cpu(),
                ),
                LOG / "phase44-layer-mismatch.pt",
            )
            raise
        work = {f"S{s}": int((ws.metrics[:, 15 + s] > 0).sum()) for s in (1, 3, 4, 6)}
        for stage, n in work.items():
            require(
                n == grid,
                stage + " active CTA work",
                ref,
                ("all CTAs",),
                count=n,
                grid=grid,
            )
        record = dict(
            seed=a.seed,
            context=a.context,
            left=a.left,
            rank=rank,
            bitwise=True,
            work_ctas=work,
            resources=list(ws.admitted.values()),
        )
        records.append(record)
        if a.replays:
            delay = torch.tensor([a.seed], device="cuda", dtype=torch.int32)
            cache = (
                f.cache_k.view(torch.uint8).clone(),
                f.cache_v.view(torch.uint8).clone(),
            )
            for i in range(a.replays):
                delay.fill_(a.seed + 7919 * i)
                layer.prepare_peers(ws, old["partial3"], old["partial5"], rank=rank)
                layer.launch(f.x, f.stream, w, m, ws, rank=rank, seed=delay)
                torch.cuda.synchronize()
                verify(f, ws, old, rank)
                for n, v, b in zip(
                    ("cache_k", "cache_v"), (f.cache_k, f.cache_v), cache
                ):
                    exact_check(v.view(torch.uint8), b, f"replay{i} entire {n}", ref)
            record["replays"] = a.replays
            print(
                json.dumps(
                    dict(event="replays", rank=rank, count=a.replays, bitwise=True)
                ),
                flush=True,
            )
        if a.timing:
            record["timing"] = {
                str(prefetch): timing(f, w, m, ws, old, rank, prefetch)
                for prefetch in (False, True)
            }
            base = json.loads(
                (LOG / f"phase43-stage-sums-{a.context}-{a.left}.json").read_text()
            )[rank]
            record["standalone_sum_us"] = base["standalone_sum_us"]
            record["native_sum_us"] = base["native_fast_sum_us"]
            record["standalone_ratio"] = (
                record["timing"]["True"]["median_us"] / record["standalone_sum_us"]
            )
        if a.profile:
            trace = torch.zeros((grid, 32), device="cuda", dtype=torch.uint64)
            waits = torch.zeros((grid, 16, 4), device="cuda", dtype=torch.uint64)
            expert = torch.zeros_like(trace)
            tasks = torch.zeros((4096, 32), device="cuda", dtype=torch.uint64)
            rt = torch.zeros((3, grid, 32), device="cuda", dtype=torch.uint64)
            layer.prepare_peers(ws, old["partial3"], old["partial5"], rank=rank)
            layer.launch(
                f.x,
                f.stream,
                w,
                m,
                ws,
                rank=rank,
                trace=trace,
                wait_trace=waits,
                expert_trace=expert,
                task_trace=tasks,
                row_trace=rt,
            )
            torch.cuda.synchronize()
            verify(f, ws, old, rank)
            torch.save(
                dict(
                    trace=trace.cpu(),
                    waits=waits.cpu(),
                    experts=expert.cpu(),
                    tasks=tasks.cpu(),
                    rows=rt.cpu(),
                    metrics=ws.metrics.cpu(),
                    num_post=ws.s4.num_post.cpu(),
                ),
                LOG / f"phase44-profile-{a.left}-r{rank}.pt",
            )
        if a.missing:
            for stage in range(1, 7):
                bad = layer.make_workspace(
                    "cuda", context=a.context, left=a.left, variant=3
                )
                bad.buffers["out"].fill_(float("nan"))
                bad.buffers["stream"].fill_(float("nan"))
                layer.prepare_peers(bad, old["partial3"], old["partial5"], rank=rank)
                layer.launch(f.x, f.stream, w, m, bad, rank=rank, skip_stage=stage)
                torch.cuda.synchronize()
                diag = layer.diagnostic_records(bad)
                require(
                    bool(diag) and not bool(bad.status[:6].eq(1).all()),
                    "missing-producer status",
                    ref,
                    (stage - 1, 0),
                )
                for n in ("out", "stream"):
                    exact_check(
                        bad.buffers[n].isnan(), True, n + " missing NaN sentinel", ref
                    )
                print(
                    json.dumps(
                        dict(
                            event="missing",
                            rank=rank,
                            stage=stage,
                            status=bad.status[:6].sum(1).tolist(),
                            diagnostics=diag[:4],
                        )
                    ),
                    flush=True,
                )
                del bad
        print(json.dumps(record), flush=True)
        suffix = "-missing" if a.missing else "-profile" if a.profile else ""
        (LOG / f"phase44-check-{a.seed}-{a.context}-{a.left}{suffix}.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        del ws
    print("PASS", a.seed, a.context, a.left, flush=True)


if __name__ == "__main__":
    main()

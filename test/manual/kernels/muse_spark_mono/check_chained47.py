"""Chained six-dispatch layer: archive matrix, graph replay, bounded failure propagation."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import (
    exact_check,
)
from reference_archive import checked_load, gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import chained, tlx_experts

LOG = Path("/home/bangtliu/coord/logs")
BASE = Path("/home/bangtliu/coord/phase44-baseline")


def archive_check(f, ws, old, rank, digest=True):
    ref = {"_rank": rank}
    exact_check(ws.failure[0], 0, "chain failure", ref)
    for s, status in enumerate(chained.statuses(ws), 1):
        exact_check(status, 1, f"S{s} status", ref)
    exact_check(ws.s2.status, 1, "S2 merge status", ref)
    exact_check(ws.s4.ids, old["ids"], "routing IDs", ref)
    # With fp8 expert activations (MX13; FOLD13 is the exact two-term form) the expert sums are
    # taken in another order than the archive's bf16 path: everything up to the experts' input
    # stays bit-exact, the experts' output and what follows it is checked numerically.
    numeric = bool(tlx_experts.MX13)
    limit = 1e-2 if tlx_experts.FOLD13 else 1.5e-1
    for name, t in old["buffers"].items():
        a = ws.buffers[name]
        typ = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
        if numeric and name in ("x6", "out", "stream"):
            err = float((a.float() - t.float()).abs().max() / t.float().abs().max())
            if not err <= limit:
                raise AssertionError(dict(event="checker-failure", tensor=name + " numeric",
                                          rank=rank, max_rel_to_max=err, limit=limit))
            continue
        exact_check(a.view(typ), t.view(typ), name + " bits", ref)
    exact_check(
        ws.s4.scores.view(torch.int32),
        old["scores"].view(torch.int32),
        "scores bits",
        ref,
    )
    exact_check(
        ws.s4.pooled.view(torch.int16), old["pool"].view(torch.int16), "pool bits", ref
    )
    slots = f.mapping[f.locations] if ws.left >= 0 else f.locations
    for name in ("cache_k", "cache_v"):
        cache = getattr(f, name).view(torch.uint8)
        exact_check(
            cache.reshape(-1, 4, 64)[slots],
            old["cache"][name],
            name + " new bytes",
            ref,
        )
        if digest:
            digest_value = hashlib.sha256(cache.cpu().numpy().tobytes()).hexdigest()
            assert digest_value == old["cache_sha256"][name], dict(
                tensor=name,
                rank=rank,
                location="whole-cache SHA256",
                actual=digest_value,
                expected=old["cache_sha256"][name],
            )


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    for n, d in [
        ("seed", 7),
        ("context", 8192),
        ("left", -1),
        ("rank", 0),
        ("replays", 3),
    ]:
        p.add_argument("--" + n, type=int, default=d)
    p.add_argument("--guard-mode", type=int, default=0, choices=(0, 1))
    p.add_argument("--log-prefix", default="phase47")
    p.add_argument("--all-ranks", action="store_true")
    p.add_argument("--missing", action="store_true")
    a = p.parse_args()
    f, w, m = make(a.seed, a.context, a.left, a.rank)
    saved = gpu(
        torch.load(BASE / f"layer-{a.seed}-{a.context}-{a.left}.pt", weights_only=True)
    )
    baseweights = {n: w[n] for n in ("w13", "w13_scale", "w2", "w2_scale")}
    ow = checked_load(f"S3-inputs-{a.seed}.pt")["weight"]
    records = []
    for rank in range(4) if a.all_ranks else (a.rank,):
        w["o_proj"] = ow[rank]
        for n, t in baseweights.items():
            w[n] = t.roll((rank - a.rank) * 11, 0).contiguous()
        old = saved["rows"][rank]
        ws = chained.make_workspace("cuda", context=a.context, left=a.left)
        chained.prepare_peers(ws, old["partial3"], old["partial5"], rank=rank)
        chained.launch(f.x, f.stream, w, m, ws, rank=rank, guard_mode=a.guard_mode)
        torch.cuda.synchronize()
        # Identical stage arithmetic should preserve the accepted archived bits.
        archive_check(f, ws, old, rank)
        print(
            json.dumps(
                dict(
                    event="archive-exact",
                    seed=a.seed,
                    context=a.context,
                    left=a.left,
                    rank=rank,
                    resources=chained.resources(ws),
                )
            ),
            flush=True,
        )
        baseline = {n: t.clone() for n, t in ws.buffers.items()}
        ids = ws.s4.ids.clone()
        delay = torch.tensor([a.seed], device="cuda", dtype=torch.int32)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chained.prepare_peers(ws, old["partial3"], old["partial5"], rank=rank)
            chained.launch(
                f.x, f.stream, w, m, ws, rank=rank, seed=delay, guard_mode=a.guard_mode
            )
        for replay in range(a.replays):
            delay.fill_(a.seed + 7919 * replay)
            graph.replay()
            torch.cuda.synchronize()
            archive_check(f, ws, old, rank)
        row = dict(
            seed=a.seed,
            context=a.context,
            left=a.left,
            rank=rank,
            archive_exact=True,
            replays=a.replays,
            resources=chained.resources(ws),
        )
        if a.missing:
            row["missing"] = []
            for stage in range(1, 7):
                bad = chained.make_workspace("cuda", context=a.context, left=a.left)
                bad.buffers["out"].fill_(float("nan"))
                bad.buffers["stream"].fill_(float("nan"))
                failed = torch.cuda.CUDAGraph()
                # Warm compile without launching a missing-producer graph yet.
                # Public API admission occurs during capture, as prior warm normal call
                # has compiled every default stage; skip-specializations compile here.
                with torch.cuda.graph(failed):
                    chained.prepare_peers(
                        bad, old["partial3"], old["partial5"], rank=rank
                    )
                    chained.launch(
                        f.x,
                        f.stream,
                        w,
                        m,
                        bad,
                        rank=rank,
                        skip_stage=stage,
                        guard_mode=a.guard_mode,
                    )
                failed.replay()
                torch.cuda.synchronize()
                ref = {"_rank": rank}
                exact_check(bad.failure[0], stage, "missing producer stage", ref)
                for later, status in enumerate(chained.statuses(bad), 1):
                    if later > stage:
                        exact_check(status, 0, f"S{later} blocked status", ref)
                for name in ("out", "stream"):
                    exact_check(
                        bad.buffers[name].isnan(), True, name + " missing sentinel", ref
                    )
                detail = dict(
                    stage=stage,
                    first_pid=int(bad.failure[1]),
                    status=[int(s.eq(1).sum()) for s in chained.statuses(bad)],
                )
                row["missing"].append(detail)
                print(
                    json.dumps(dict(event="missing", rank=rank, **detail)), flush=True
                )
        records.append(row)
        (
            LOG / f"{a.log_prefix}-chain-check-{a.seed}-{a.context}-{a.left}.json"
        ).write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(dict(event="case-done", **row)), flush=True)


if __name__ == "__main__":
    main()

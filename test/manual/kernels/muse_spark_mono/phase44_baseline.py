"""Immutable GPU0 outputs of the accepted layer, for phase4.4 bitwise checks."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import build, compare, routing_check, teacher_reference
from reference_archive import checked_load, cpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer

ROOT = Path("/home/bangtliu/coord/phase44-baseline")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    for name, default in (("seed", 7), ("context", 8192), ("left", -1)):
        p.add_argument("--" + name, type=int, default=default)
    a = p.parse_args()
    path = ROOT / f"layer-{a.seed}-{a.context}-{a.left}.pt"
    if path.exists():
        raise FileExistsError(path)
    f, w, m = make(a.seed, a.context, a.left, 0)
    base = build(f, w, a.seed, a.context, a.left, expert_fp64=False)
    wo = checked_load(f"S3-inputs-{a.seed}.pt")["weight"]
    experts = {n: w[n] for n in ("w13", "w13_scale", "w2", "w2_scale")}
    rows = []
    for rank in range(4):
        ref = dict(base, _rank=rank)
        w["o_proj"] = wo[rank]
        for n, t in experts.items():
            w[n] = t.roll(rank * 11, 0).contiguous()
        ws = layer.make_workspace("cuda", context=a.context, left=a.left)
        layer.prepare_peers(ws, ref["partial3"], ref["partial5"], rank=rank)
        layer.launch(f.x, f.stream, w, m, ws, rank=rank)
        torch.cuda.synchronize()
        routes = routing_check(ws, ref, w)
        if any(not row["set_equal"] for row in routes):
            ref = teacher_reference(f, w, ref, ws, rank, routes)
        errors = compare(f, ws, ref, w)
        slots = f.mapping[f.locations] if a.left >= 0 else f.locations
        cache = {
            n: getattr(f, n).view(torch.uint8).reshape(-1, 4, 64)[slots].cpu()
            for n in ("cache_k", "cache_v")
        }
        digests = {
            n: hashlib.sha256(
                getattr(f, n).view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()
            for n in ("cache_k", "cache_v")
        }
        rows.append(
            cpu(
                dict(
                    rank=rank,
                    buffers={
                        n: t for n, t in ws.buffers.items() if not n.startswith("f")
                    },
                    ids=ws.s4.ids,
                    scores=ws.s4.scores,
                    pool=ws.s4.pooled,
                    partial3=ref["partial3"],
                    partial5=ref["partial5"],
                    cache=cache,
                    cache_sha256=digests,
                    errors=errors,
                    teacher_rows=ref.get("teacher_rows", []),
                )
            )
        )
        print(
            json.dumps(
                dict(
                    event="baseline-rank",
                    seed=a.seed,
                    context=a.context,
                    left=a.left,
                    rank=rank,
                )
            ),
            flush=True,
        )
        del ws
    torch.save(dict(seed=a.seed, context=a.context, left=a.left, rows=rows), path)
    manifest = ROOT / "manifest.json"
    d = (
        json.loads(manifest.read_text())
        if manifest.exists()
        else dict(commit="0dd70dd0f7a3b7f61181a9c2abaee573b838d0f6", artifacts={})
    )
    d["artifacts"][path.name] = dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size
    )
    manifest.write_text(json.dumps(d, indent=2) + "\n")
    print("PASS baseline", path, flush=True)


if __name__ == "__main__":
    main()

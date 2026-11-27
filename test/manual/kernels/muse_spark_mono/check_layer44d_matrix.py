"""Exact48-case archive comparison after the local4.4d row epilogue fix."""

import argparse
import hashlib
import importlib.util
import json
import sys

import torch
from check_layer44 import BASE, LOG, verify
from layer_case import make
from layer_fixture import require_gpu0
from reference_archive import checked_load, gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer


def main():
    require_gpu0()
    spec = importlib.util.spec_from_file_location(
        "layer_packed44d", BASE.parent / "layer_packed44d.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    layer.layer_accepted._launch = mod._launch
    p = argparse.ArgumentParser()
    for n in ("seed", "context", "left"):
        p.add_argument("--" + n, type=int, required=True)
    a = p.parse_args()
    path = BASE / f"layer-{a.seed}-{a.context}-{a.left}.pt"
    manifest = json.loads((BASE / "manifest.json").read_text())
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == manifest["artifacts"][path.name]["sha256"]
    )
    old = torch.load(path, weights_only=True)
    f, w, m = make(a.seed, a.context, a.left, 0)
    wo = checked_load(f"S3-inputs-{a.seed}.pt")["weight"]
    ew = {n: w[n] for n in ("w13", "w13_scale", "w2", "w2_scale")}
    records = []
    for rank in range(4):
        w["o_proj"] = wo[rank]
        for n, v in ew.items():
            w[n] = v.roll(rank * 11, 0).contiguous()
        ref = gpu(old["rows"][rank])
        ws = layer.make_workspace("cuda", context=a.context, left=a.left)
        k = layer.launch(f.x, f.stream, w, m, ws, rank=rank, compile_only=True)
        ws.shape.admit("layer44d-matrix", k, grid=768)
        layer.prepare_peers(ws, ref["partial3"], ref["partial5"], rank=rank)
        layer.launch(f.x, f.stream, w, m, ws, rank=rank)
        torch.cuda.synchronize()
        try:
            verify(f, ws, ref, rank, True)
        except AssertionError:
            torch.save(
                dict(
                    rank=rank,
                    buffers={n: t.cpu() for n, t in ws.buffers.items()},
                    ids=ws.s4.ids.cpu(),
                    scores=ws.s4.scores.cpu(),
                    pool=ws.s4.pooled.cpu(),
                ),
                LOG / "phase44d-layer-mismatch.pt",
            )
            raise
        rec = dict(
            seed=a.seed,
            context=a.context,
            left=a.left,
            rank=rank,
            bitwise=True,
            vgpr=k.n_regs,
            spills=k.n_spills,
            lds=k.metadata.shared,
        )
        records.append(rec)
        print(json.dumps(rec), flush=True)
        del ws
    (LOG / f"phase44d-matrix-{a.seed}-{a.context}-{a.left}.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

"""Diagnostic-only matched snapshots of accepted and tiled layer handoffs."""

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
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--accepted", action="store_true")
    p.add_argument("--light", action="store_true")
    p.add_argument("--counts", action="store_true")
    a = p.parse_args()
    suffix = "-counts" if a.counts else ("-light" if a.light else "")
    out = (
        LOG
        / f"phase44b-profile-{a.context}-{a.left}-{'accepted' if a.accepted else 'tiled'}{suffix}.pt"
    )
    if out.exists():
        raise FileExistsError(out)
    mod = layer
    if a.accepted:
        spec = importlib.util.spec_from_file_location(
            "layer_phase42",
            (BASE / "layer_phase42.py")
            if a.light
            else (BASE.parent / "layer_phase42_diagnostic.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    path = BASE / f"layer-7-{a.context}-{a.left}.pt"
    manifest = json.loads((BASE / "manifest.json").read_text())
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == manifest["artifacts"][path.name]["sha256"]
    )
    old = gpu(torch.load(path, weights_only=True)["rows"][0])
    f, w, m = make(7, a.context, a.left, 0)
    w["o_proj"] = checked_load("S3-inputs-7.pt")["weight"][0]
    ws = mod.make_workspace(
        "cuda", context=a.context, left=a.left, **({} if a.accepted else {"variant": 3})
    )
    grid = ws.shape.grid("cuda")
    trace = torch.zeros((grid, 32), device="cuda", dtype=torch.uint64)
    waits = torch.zeros((grid, 16, 4), device="cuda", dtype=torch.uint64)
    expert = torch.zeros_like(trace)
    tasks = torch.zeros((4096, 32), device="cuda", dtype=torch.uint64)
    rows = torch.zeros((3, grid, 32), device="cuda", dtype=torch.uint64)
    pt = torch.zeros((6, 4096, 32), device="cuda", dtype=torch.uint64)
    opts = dict(
        rank=0,
        trace=trace,
        wait_trace=waits,
        expert_trace=expert,
        task_trace=tasks,
        profile_tasks=pt,
    )
    if a.light:
        opts.pop("profile_tasks")
    if not a.accepted:
        opts.update(row_trace=rows)
    k = mod.launch(f.x, f.stream, w, m, ws, compile_only=True, **opts)
    resource = ws.shape.admit("phase44b-diagnostics", k, grid=grid)
    print(
        json.dumps(
            dict(
                event="admitted",
                accepted=a.accepted,
                vgpr=k.n_regs,
                lds=k.metadata.shared,
                spills=k.n_spills,
                occupancy=resource.active_ctas_per_cu,
            )
        ),
        flush=True,
    )
    # One diagnostic snapshot; no tuning or graph-time comparison with instrumented binaries.
    mod.prepare_peers(ws, old["partial3"], old["partial5"], rank=0)
    mod.launch(f.x, f.stream, w, m, ws, **opts)
    torch.cuda.synchronize()
    verify(f, ws, old, 0, True)
    data = dict(
        context=a.context,
        left=a.left,
        accepted=a.accepted,
        bitwise=True,
        trace=trace.cpu(),
        waits=waits.cpu(),
        experts=expert.cpu(),
        tasks=tasks.cpu(),
        rows=rows.cpu(),
        profile_tasks=pt.cpu(),
        metrics=ws.metrics.cpu(),
        num_post=ws.s4.num_post.cpu(),
        epochs=ws.epochs.cpu(),
        flags=ws.flags.cpu(),
        sorted_ids=ws.s4.sorted_ids.cpu(),
        expert_ids=ws.s4.expert_ids.cpu(),
        resources=dict(vgpr=k.n_regs, lds=k.metadata.shared, spills=k.n_spills),
    )
    if not a.accepted:
        data.update(row3=ws.row3.cpu(), row4=ws.row4.cpu(), row6=ws.row6.cpu())
    torch.save(data, out)
    print(
        json.dumps(
            dict(
                event="bitwise_pass",
                path=str(out),
                status=ws.status[:6].sum(1).tolist(),
                num_post=ws.s4.num_post.item(),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

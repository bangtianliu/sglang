"""Bounded Phase4.4d variant admission, exact archived bits, timing and light trace."""

import argparse
import json

import torch
from check_layer44 import BASE, LOG, timing, verify
from layer_case import make
from layer_fixture import require_gpu0
from reference_archive import checked_load, gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=int, required=True, choices=(0, 1, 2))
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--profile-only", action="store_true")
    p.add_argument("--control", action="store_true")
    a = p.parse_args()
    label = "control" if a.control else "fixed"
    import importlib.util
    import sys

    for name, path in (
        [
            ("boundary_before44d", BASE.parent / "phase44d-boundary-before.py"),
            ("layer_before44d", BASE.parent / "layer_before44d.py"),
        ]
        if a.control
        else [("layer_packed44d", BASE.parent / "layer_packed44d.py")]
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    layer.layer_accepted._launch = mod._launch
    old = gpu(torch.load(BASE / "layer-7-8192--1.pt", weights_only=True)["rows"][0])
    f, w, m = make(7, 8192, -1, 0)
    w["o_proj"] = checked_load("S3-inputs-7.pt")["weight"][0]
    ws = layer.make_workspace("cuda", context=8192, left=-1, variant=a.variant)
    k = layer.launch(f.x, f.stream, w, m, ws, rank=0, compile_only=True)
    admit = ws.shape.admit("phase44c", k, grid=768)
    print(
        json.dumps(
            dict(
                event="admit",
                variant=a.variant,
                vgpr=k.n_regs,
                lds=k.metadata.shared,
                spills=k.n_spills,
                occupancy=admit.active_ctas_per_cu,
            )
        ),
        flush=True,
    )
    for fmt in ("llir", "amdgcn"):
        (LOG / f"phase44d-layer-{label}.{fmt}").write_text(k.asm[fmt])
    if a.compile_only:
        return
    if not a.profile_only:
        layer.prepare_peers(ws, old["partial3"], old["partial5"], rank=0)
        layer.launch(f.x, f.stream, w, m, ws, rank=0)
        torch.cuda.synchronize()
        print(
            json.dumps(
                dict(
                    event="status",
                    status=ws.status[:6].sum(1).tolist(),
                    diagnostics=layer.diagnostic_records(ws)[:8],
                )
            ),
            flush=True,
        )
        try:
            verify(f, ws, old, 0, True)
        except AssertionError:
            torch.save(
                dict(
                    buffers={n: t.cpu() for n, t in ws.buffers.items()},
                    row3=ws.row3.cpu(),
                    row4=ws.row4.cpu(),
                    row6=ws.row6.cpu(),
                    ids=ws.s4.ids.cpu(),
                    scores=ws.s4.scores.cpu(),
                ),
                LOG / f"phase44d-layer-{label}-mismatch.pt",
            )
            raise
        print(json.dumps(dict(event="bitwise_pass", variant=a.variant)), flush=True)
        record = dict(
            variant=a.variant, bitwise=True, timing=timing(f, w, m, ws, old, 0, True)
        )
        print(json.dumps(record), flush=True)
    else:
        record = json.loads((LOG / f"phase44d-layer-{label}.json").read_text())
    trace = torch.zeros((768, 32), device="cuda", dtype=torch.uint64)
    waits = torch.zeros((768, 16, 4), device="cuda", dtype=torch.uint64)
    rows = torch.zeros((3, 768, 32), device="cuda", dtype=torch.uint64)
    tasks = torch.zeros((6, 4096, 32), device="cuda", dtype=torch.uint64)
    opts = dict(rank=0, trace=trace, wait_trace=waits)
    if a.variant:
        opts.update(row_trace=rows, profile_tasks=tasks, task_times=False)
    kp = layer.launch(f.x, f.stream, w, m, ws, compile_only=True, **opts)
    ws.shape.admit("phase44c-light", kp, grid=768)
    layer.prepare_peers(ws, old["partial3"], old["partial5"], rank=0)
    layer.launch(f.x, f.stream, w, m, ws, **opts)
    torch.cuda.synchronize()
    verify(f, ws, old, 0, True)
    torch.save(
        dict(
            trace=trace.cpu(),
            waits=waits.cpu(),
            rows=rows.cpu(),
            tasks=tasks.cpu(),
            metrics=ws.metrics.cpu(),
            epochs=ws.epochs.cpu(),
        ),
        LOG / f"phase44d-layer-{label}-profile-light.pt",
    )
    (LOG / f"phase44d-layer-{label}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

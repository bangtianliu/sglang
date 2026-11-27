"""Interleaved old/fixed CUDA-graph timing with identical inputs and peer setup."""

import importlib.util
import json
import sys

import torch
from check_layer44 import BASE, LOG, verify
from layer_case import make
from layer_fixture import require_gpu0
from reference_archive import checked_load, gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer


def distribution(samples):
    ordered = sorted(samples)
    n = len(ordered)
    return dict(
        n=n,
        median_us=(ordered[(n - 1) // 2] + ordered[n // 2]) / 2,
        p99_us=ordered[max(0, (99 * n + 99) // 100 - 1)],
    )


def main():
    output = LOG / "phase44d-layer-paired-orders.json"
    if output.exists():
        raise FileExistsError(output)
    require_gpu0()
    accepted = layer.layer_accepted._launch
    for name, path in [
        ("boundary_before44d", BASE.parent / "phase44d-boundary-before.py"),
        ("layer_before44d", BASE.parent / "layer_before44d.py"),
        ("layer_packed44d", BASE.parent / "layer_packed44d.py"),
    ]:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    funcs = {
        "control": sys.modules["layer_before44d"]._launch,
        "fixed": sys.modules["layer_packed44d"]._launch,
    }
    old = gpu(torch.load(BASE / "layer-7-8192--1.pt", weights_only=True)["rows"][0])
    f, w, m = make(7, 8192, -1, 0)
    w["o_proj"] = checked_load("S3-inputs-7.pt")["weight"][0]
    runs = {}
    for label, fn in funcs.items():
        layer.layer_accepted._launch = fn
        ws = layer.make_workspace("cuda", context=8192, left=-1)
        layer.prepare_peers(ws, old["partial3"], old["partial5"], rank=0)
        layer.launch(f.x, f.stream, w, m, ws, rank=0)
        torch.cuda.synchronize()
        verify(f, ws, old, 0, True)
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer.prepare_peers(ws, old["partial3"], old["partial5"], rank=0)
            start.record()
            layer.launch(f.x, f.stream, w, m, ws, rank=0)
            end.record()
        runs[label] = (
            ws,
            graph,
            start,
            end,
            [],
            {"control_then_fixed": [], "fixed_then_control": []},
        )
    layer.layer_accepted._launch = accepted
    for rep in range(210):
        order = "control_then_fixed" if rep % 2 == 0 else "fixed_then_control"
        for label in ("control", "fixed") if rep % 2 == 0 else ("fixed", "control"):
            ws, g, start, end, us, orders = runs[label]
            g.replay()
            torch.cuda.synchronize()
            assert bool(ws.status[:6].eq(1).all()), (
                label,
                rep,
                layer.diagnostic_records(ws),
            )
            if rep >= 10:
                elapsed = start.elapsed_time(end) * 1000
                us.append(elapsed)
                orders[order].append(elapsed)
    result = {}
    for label, (ws, g, start, end, us, orders) in runs.items():
        verify(f, ws, old, 0, True)
        assert len(us) >= 200 and all(len(v) >= 100 for v in orders.values())
        result[label] = dict(
            **distribution(us),
            by_order={k: distribution(v) for k, v in orders.items()},
            samples_by_order_us=orders,
            bitwise=True,
            resources=list(ws.admitted.values()),
        )
    result["speedup"] = result["control"]["median_us"] / result["fixed"]["median_us"]
    print(json.dumps(result), flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

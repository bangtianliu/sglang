"""One full/sliding GPU0 layer timeline; instrumentation is absent in normal binaries."""

import argparse
import json
from pathlib import Path

import torch
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import build, compare

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=int, default=-1)
    a = p.parse_args()
    f, w, m = make(7, 131072, a.left, 0)
    ref = build(f, w, 7, 131072, a.left, expert_fp64=False)
    ref["_rank"] = 0
    ws = layer.make_workspace("cuda", context=131072, left=a.left)
    grid = ws.shape.grid("cuda")
    trace = torch.zeros((grid, 32), device="cuda", dtype=torch.uint64)
    waits = torch.zeros((grid, 16, 4), device="cuda", dtype=torch.uint64)
    experts = torch.zeros((grid, 32), device="cuda", dtype=torch.uint64)
    tasks = torch.zeros((4096, 32), device="cuda", dtype=torch.uint64)
    layer.prepare_peers(ws, ref["partial3"], ref["partial5"], rank=0)
    layer.launch(
        f.x,
        f.stream,
        w,
        m,
        ws,
        rank=0,
        trace=trace,
        wait_trace=waits,
        expert_trace=experts,
        task_trace=tasks,
    )
    torch.cuda.synchronize()
    assert ws.status[:6].eq(1).all()
    errors = compare(f, ws, ref, w)
    assert ws.metrics[:, 31].max() <= 16
    root = Path("/home/bangtliu/coord/logs")
    stem = f"phase43-layer-profile-{a.left}"
    torch.save(
        dict(
            trace=trace.cpu(),
            waits=waits.cpu(),
            experts=experts.cpu(),
            tasks=tasks.cpu(),
            metrics=ws.metrics.cpu(),
            num_post=ws.s4.num_post.cpu(),
        ),
        root / f"{stem}.pt",
    )
    row = dict(
        left=a.left,
        context=131072,
        rank=0,
        resources=list(ws.admitted.values()),
        errors=errors,
        max_wait_events=int(ws.metrics[:, 31].max()),
    )
    (root / f"{stem}.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

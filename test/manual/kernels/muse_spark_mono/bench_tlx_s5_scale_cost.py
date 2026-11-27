"""Isolate scale-copy substitution in the unchanged Gluon expert math/core.

Generate an experimental local core copy under coord/logs, replacing only
scale async copies with gl.load + LDS store. Reference source is untouched.
Three distinct weight copies; same-process alternating AB/BA; exact outputs.
This measures substitution inside a real core, not cycles attributed to TLX.
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch
from check_experts import cases
from check_experts_complete import table_for
from layer_fixture import require_gpu0
from layer_reference import exact_check
from paired45 import measure
from reference_archive import checked_load
from tlx_ir import dump
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    experts_async as original,
)

ROOT = Path("/home/bangtliu/coord/logs")
source = Path(original.__file__).read_text()
needle = "gfx1250.async_copy.global_to_shared("
pos = 0
count = 0
while True:
    start = source.find(needle, pos)
    if start < 0:
        break
    a = start + len(needle)
    depth = 1
    i = a
    while depth:
        depth += (source[i] == "(") - (source[i] == ")")
        i += 1
    args = source[a : i - 1]
    depth = 0
    split = None
    for j, c in enumerate(args):
        depth += (c == "(") - (c == ")")
        if c == "," and depth == 0:
            split = j
            break
    dst, src = args[:split].strip(), args[split + 1 :].strip()
    if dst.startswith(("ss.index", "st.index")):
        new = dst + ".store(gl.load(" + src + "))"
        source = source[:start] + new + source[i:]
        pos = start + len(new)
        count += 1
    else:
        pos = i
assert count == 4, count
path = ROOT / "phaseT3c-scale-sync-core.py"
path.write_text(source)
spec = importlib.util.spec_from_file_location("t3c_scale_sync_core", path)
scale_sync = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scale_sync
spec.loader.exec_module(scale_sync)


@g.jit
def launch(
    X, Sorted, Experts, NumPost, W, Scale, Out, FIRST: gl.constexpr, SYNC: gl.constexpr
):
    pid = gl.program_id(0)
    for task in range(pid, gl.load(NumPost) // 16 * 64, 1728):
        block = task // 64
        expert = gl.load(Experts + block)
        if SYNC:
            scale_sync.expert_tile(
                X,
                Sorted,
                W,
                Scale,
                Out,
                block,
                expert,
                task % 64,
                FIRST,
                not FIRST,
                READ_MODE=2,
                SCOPED_OUTPUT=False,
            )
        else:
            original.expert_tile(
                X,
                Sorted,
                W,
                Scale,
                Out,
                block,
                expert,
                task % 64,
                FIRST,
                not FIRST,
                READ_MODE=2,
                SCOPED_OUTPUT=False,
            )


def main():
    require_gpu0()
    saved = checked_load("phase3e1-inputs-7.pt")
    ids = cases(7, "cuda")["typical"]
    tables = table_for(ids)
    table = tuple(v.reshape(4, -1)[0] for v in tables)
    copies = [{k: v.clone() for k, v in saved["weights"].items()} for _ in range(3)]
    inputs = [saved["x"].clone() for _ in copies]
    mid = [torch.empty((64, 4096), device="cuda", dtype=torch.bfloat16) for _ in copies]
    outputs = {s: [torch.empty_like(mid[0]) for _ in copies] for s in (False, True)}
    results = {}
    for first in (True, False):

        def call(sync):
            for i, w in enumerate(copies):
                k = launch[(1728,)](
                    inputs[i] if first else mid[i],
                    *table,
                    w["w13" if first else "w2"],
                    w["w13_scale" if first else "w2_scale"],
                    outputs[sync][i],
                    FIRST=first,
                    SYNC=sync,
                    num_warps=4,
                    waves_per_eu=9,
                    enable_fp_fusion=False,
                )
            return k

        kernels = {s: call(s) for s in (False, True)}
        torch.cuda.synchronize()
        valid = ids.flatten() < 64
        for i in range(3):
            exact_check(
                outputs[True][i][valid].view(torch.int16),
                outputs[False][i][valid].view(torch.int16),
                "scale substitution " + ("W13" if first else "W2"),
                {"_rank": 0},
            )

        def check():
            for i in range(3):
                exact_check(
                    outputs[True][i][valid].view(torch.int16),
                    outputs[False][i][valid].view(torch.int16),
                    "scale substitution",
                    {"_rank": 0},
                )

        cases_ = {
            label: (lambda: None, lambda s=s: call(s), check)
            for label, s in [("async", False), ("normal", True)]
        }
        result, _ = measure(cases_)
        ratios = {
            o: result["normal"]["by_order"][o]["median_us"]
            / result["async"]["by_order"][o]["median_us"]
            for o in ("AB", "BA")
        }
        name = "w13" if first else "w2"
        for s, k in kernels.items():
            dump(k, "T3c-S5-scale-cost/" + name, "normal" if s else "async")
        results[name] = dict(
            result=result,
            ratios=ratios,
            copies=3,
            per_copy_median_us={
                label: {o: r["by_order"][o]["median_us"] / 3 for o in ("AB", "BA")}
                for label, r in result.items()
            },
        )
        print(json.dumps(dict(phase=name, **results[name])), flush=True)
        if first:
            for i in range(3):
                mid[i].copy_(outputs[False][i])
    (ROOT / "phaseT3c-S5-scale-cost.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

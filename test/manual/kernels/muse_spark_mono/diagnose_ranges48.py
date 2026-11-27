"""Dependent graph stage ranges vs regular; capture configs without profiling tools."""

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from bench_chained47 import setup
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from paired45 import distribution
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import chained

LOG = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--prefix", default="phase48")
    a = p.parse_args()
    cases, native, ws, errors, routes = setup(a.context, a.left)
    events = {
        n: [torch.cuda.Event(enable_timing=True, external=True) for _ in range(7)]
        for n in cases
    }
    current = [1]
    configs = {}
    orig_metadata = CompiledKernel.launch_metadata

    def metadata(kernel, *aa, **kk):
        result = orig_metadata(kernel, *aa, **kk)
        key = kernel.name + ":" + kernel.hash
        configs.setdefault(str(current[0]), {})[key] = dict(
            name=kernel.name,
            module=kernel.src.fn.fn.__module__,
            metadata={k: str(v) for k, v in kernel.metadata._asdict().items()},
            constants={str(k): str(v) for k, v in kernel.src.constants.items()},
            arg_names=list(kernel.src.fn.arg_names),
            vgpr=kernel.n_regs,
            spills=kernel.n_spills,
            sha256=hashlib.sha256(kernel.asm["hsaco"]).hexdigest(),
        )
        return result

    def mark_native(stage):
        events["regular"][stage].record()
        current[0] = stage + 1

    def native_call():
        current[0] = 1
        events["regular"][0].record()
        return native(mark=mark_native)

    def chain_call():
        events["chained"][0].record()
        with ExitStack() as stack:
            for stage, (module, name) in enumerate(
                [
                    (chained.s1, "qkvg"),
                    (chained.s2, "attention"),
                    (
                        chained.independent_attn_tail
                        if ws.stage_policy[0] == "wide"
                        else chained.s3,
                        "attention_tail",
                    ),
                    (chained.s4, "routing"),
                    (
                        chained.independent_experts
                        if ws.stage_policy[1] == "k9"
                        else chained.s5,
                        "s5",
                    ),
                    (
                        chained.independent_tail
                        if ws.stage_policy[2] == "wide"
                        else chained.s6,
                        "unpool_boundary",
                    ),
                ],
                1,
            ):
                fn = getattr(module, name)

                def wrapped(*aa, _fn=fn, _stage=stage, **kk):
                    result = _fn(*aa, **kk)
                    events["chained"][_stage].record()
                    return result

                stack.enter_context(patch.object(module, name, wrapped))
            return cases["chained"][1]()

    # Record native launch configurations once, before graph timing. The hook
    # only observes the compiled kernels; no profiler/counter runtime is loaded.
    native.prepare()
    with patch.object(CompiledKernel, "launch_metadata", metadata):
        native_call()
    torch.cuda.synchronize()
    cases["regular"][2]()
    profiled = {
        "regular": (cases["regular"][0], native_call, cases["regular"][2]),
        "chained": (cases["chained"][0], chain_call, cases["chained"][2]),
    }

    def observe(label):
        ee = events[label]
        return {"stages_us": [ee[i].elapsed_time(ee[i + 1]) * 1000 for i in range(6)]}

    result = balanced(
        profiled,
        {"regular": native.epochs[0], "chained": ws.s3.state.epochs},
        observer=observe,
    )
    tables = {}
    for label in cases:
        tables[label] = {
            f"S{i + 1}": dict(
                **distribution(
                    [r["stages_us"][i] for r in result["raw"] if r["label"] == label]
                ),
                by_order={
                    o: distribution(
                        [
                            r["stages_us"][i]
                            for r in result["raw"]
                            if r["label"] == label and r["order"] == o
                        ]
                    )
                    for o in ("AB", "BA")
                },
            )
            for i in range(6)
        }
    totals = {
        label: distribution(
            [sum(r["stages_us"]) for r in result["raw"] if r["label"] == label]
        )
        for label in cases
    }
    result.update(
        stage_policy=ws.stage_policy,
        context=a.context,
        left=a.left,
        stages=tables,
        range_sum=totals,
        native_config=native.cfg,
        native_kernels=configs,
        regular_errors=errors,
        routing=routes,
        resources=chained.resources(ws),
        instrumentation="seven external graph events per layer, six consecutive dependent stage ranges; no stage tuning",
    )
    (LOG / f"{a.prefix}-ranges-{a.context}-{a.left}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({"stages": tables, "range_sum": totals}), flush=True)


if __name__ == "__main__":
    main()

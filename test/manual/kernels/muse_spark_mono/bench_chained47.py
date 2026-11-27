"""True dependent six-dispatch chain vs real native layer graph, interleaved."""

import argparse
import functools
import json

import torch
from check_chained47 import BASE, LOG, archive_check
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import (
    build,
    exact_check,
    named_check,
    routing_check,
    teacher_reference,
)
from paired45 import measure
from reference_archive import gpu
from regular_layer45 import Regular

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import chained


def setup(context, left):
    f, w, m = make(7, context, left, 0)
    old = gpu(
        torch.load(BASE / f"layer-7-{context}-{left}.pt", weights_only=True)["rows"][0]
    )
    ref = build(f, w, 7, context, left, expert_fp64=False)
    ref["_rank"] = 0
    native = Regular(f, w, m, context, left, (old["partial3"], old["partial5"]))
    native.prepare()
    native()
    torch.cuda.synchronize()
    routes = routing_check(native, ref, w)
    target = (
        teacher_reference(f, w, ref, native, 0, routes)
        if any(not r["set_equal"] for r in routes)
        else ref
    )
    errors = {
        n: named_check(
            t,
            target[n],
            0.008
            if n == "attn"
            else (
                0.02
                if target.get("teacher_forced") and n in ("x6", "out", "stream")
                else 0.03
            ),
            "regular " + n,
            ref,
        )
        for n, t in native.outputs.items()
    }
    native_outputs = {n: t.clone() for n, t in native.outputs.items()}
    native_ids = native.s4.ids.clone()
    ws = chained.make_workspace("cuda", context=context, left=left)
    prep = functools.partial(
        chained.prepare_peers, ws, old["partial3"], old["partial5"], rank=0
    )
    call = functools.partial(chained.launch, f.x, f.stream, w, m, ws, rank=0)

    def check_regular():
        for i, s in enumerate(native.status):
            exact_check(s, 1, f"regular collective {i}", ref)
        exact_check(native.s4.ids, native_ids, "regular routing replay", ref)
        for n, t in native.outputs.items():
            exact_check(t, native_outputs[n], "regular " + n + " replay", ref)

    def check_chain():
        archive_check(f, ws, old, 0, digest=False)

    return (
        {
            "regular": (native.prepare, native, check_regular),
            "chained": (prep, call, check_chain),
        },
        native,
        ws,
        errors,
        routes,
    )


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    a = p.parse_args()
    cases, native, ws, errors, routes = setup(a.context, a.left)
    result, _ = measure(cases)
    result.update(
        context=a.context,
        left=a.left,
        rank=0,
        resources=chained.resources(ws),
        regular_errors=errors,
        routing=routes,
        ratio=result["chained"]["median_us"] / result["regular"]["median_us"],
        target_met=result["chained"]["median_us"] <= result["regular"]["median_us"],
        launches=6,
        bridge="S5 BF16 X6 store included in its launch",
        failure_guard="device first-failure stage checked at each dispatch; setup outside timing",
    )
    (LOG / f"phase47-chain-vs-regular-{a.context}-{a.left}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                n: {k: v for k, v in r.items() if k != "samples_by_order_us"}
                if isinstance(r, dict)
                else r
                for n, r in result.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

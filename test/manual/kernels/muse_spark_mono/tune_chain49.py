"""Per-stage candidates measured only inside the actual dependent chain."""

import argparse
import functools
import json
import random
import statistics
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from check_chained47 import BASE, archive_check
from diagnose_guard48 import balanced
from layer_case import make
from layer_fixture import require_gpu0
from reference_archive import gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    chained,
    independent_attn_tail,
    independent_experts,
    independent_tail,
)

LOG = Path("/home/bangtliu/coord/logs")


def verdict(result):
    rng = random.Random(49)
    orders = {}
    win = True
    for order in ("AB", "BA"):
        a = [
            r["us"]
            for r in result["raw"]
            if r["label"] == "baseline" and r["order"] == order
        ]
        b = [
            r["us"]
            for r in result["raw"]
            if r["label"] == "candidate" and r["order"] == order
        ]
        delta = [x - y for x, y in zip(a, b)]
        ci = sorted(
            statistics.median(rng.choices(delta, k=len(delta))) for _ in range(1000)
        )
        gain = 1 - statistics.median(b) / statistics.median(a)
        orders[order] = dict(
            reduction=gain,
            paired_delta_us=statistics.median(delta),
            paired_bootstrap95_us=[ci[25], ci[974]],
        )
        win = win and gain > 0.02 and ci[25] > 0
    return dict(
        win=bool(win),
        orders=orders,
        rule="Both order medians >2% faster and paired-bootstrap95 lower delta >0; predeclared before further trials.",
    )


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("s6", "s3", "s5"), default="s6")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--variant", choices=("plain", "packed"), default="plain")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    assert a.variant == "plain", (
        "packed investigation is isolated in a separate bounded script"
    )
    f, w, m = make(7, a.context, a.left, 0)
    old = gpu(
        torch.load(BASE / f"layer-7-{a.context}-{a.left}.pt", weights_only=True)[
            "rows"
        ][0]
    )
    spaces = {}
    cases = {}
    epochs = {}
    for name in ("baseline", "candidate"):
        ws = chained.make_workspace(
            "cuda",
            context=a.context,
            left=a.left,
            stage_policy=("common", "common", "common" if a.stage == "s6" else "wide")
            if a.stage != "s5"
            else (
                "wide"
                if (a.context, a.left) in ((8192, 2047), (131072, -1))
                else "common",
                "common",
                "wide",
            ),
        )
        if name == "candidate":
            if a.stage == "s6":
                ws.s6 = independent_tail.make_workspace("cuda")
            elif a.stage == "s3":
                ws.s3 = independent_attn_tail.make_workspace("cuda")
            else:
                ws.s5 = independent_experts.make_workspace("cuda")
        spaces[name] = ws
        epochs[name] = ws.s3.state.epochs

        def call(ws=ws, name=name):
            with ExitStack() as stack:
                if name == "candidate":
                    if a.stage == "s6":
                        stack.enter_context(
                            patch.object(
                                chained.s6,
                                "unpool_boundary",
                                independent_tail.unpool_boundary,
                            )
                        )
                    elif a.stage == "s3":
                        stack.enter_context(
                            patch.object(
                                chained.s3,
                                "attention_tail",
                                independent_attn_tail.attention_tail,
                            )
                        )
                    else:
                        stack.enter_context(
                            patch.object(chained.s5, "s5", independent_experts.s5)
                        )
                return chained.launch(f.x, f.stream, w, m, ws, rank=0)

        cases[name] = (
            functools.partial(
                chained.prepare_peers, ws, old["partial3"], old["partial5"], rank=0
            ),
            call,
            functools.partial(archive_check, f, ws, old, 0, digest=False),
        )
    result = balanced(cases, epochs)
    result["verdict"] = verdict(result)
    result.update(
        stage=a.stage,
        context=a.context,
        left=a.left,
        resources={n: chained.resources(ws) for n, ws in spaces.items()},
    )
    (LOG / f"phase49-{a.stage}-{a.context}-{a.left}{a.tag}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["timing"]), flush=True)


if __name__ == "__main__":
    main()

"""Whole-chain A/B controls for phase4.10; candidate is never adopted here."""

import argparse
import functools
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from check_chained47 import BASE, archive_check
from ci410 import block_verdict
from diagnose_guard48 import balanced
from layer_case import make
from layer_fixture import require_gpu0
from reference_archive import gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    chained,
    independent_qkvg,
    pipelined_attention,
    split_attention,
)

LOG = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--candidate", choices=("s1", "s2", "s2-split"), default="s1")
    a = p.parse_args()
    f, w, m = make(7, a.context, a.left, 0)
    old = gpu(
        torch.load(BASE / f"layer-7-{a.context}-{a.left}.pt", weights_only=True)[
            "rows"
        ][0]
    )
    cases = {}
    spaces = {}
    epochs = {}
    for name in ("baseline", "candidate"):
        ws = chained.make_workspace(
            "cuda", context=a.context, left=a.left, attention_backend="common"
        )
        if name == "candidate":
            if a.candidate == "s1":
                ws.s1 = independent_qkvg.make_workspace("cuda")
            else:
                ws.s2 = (
                    split_attention
                    if a.candidate == "s2-split"
                    else pipelined_attention
                ).make_workspace("cuda", context=a.context, window_left=a.left)
        spaces[name] = ws
        epochs[name] = ws.s3.state.epochs

        def call(ws=ws, name=name):
            with ExitStack() as stack:
                if name == "candidate":
                    if a.candidate == "s1":
                        stack.enter_context(
                            patch.object(chained.s1, "qkvg", independent_qkvg.qkvg)
                        )
                    else:
                        stack.enter_context(
                            patch.object(
                                chained.s2,
                                "attention",
                                (
                                    split_attention
                                    if a.candidate == "s2-split"
                                    else pipelined_attention
                                ).attention,
                            )
                        )
                return chained.launch(f.x, f.stream, w, m, ws, rank=0)

        cases[name] = (
            functools.partial(
                chained.prepare_peers, ws, old["partial3"], old["partial5"], rank=0
            ),
            call,
            functools.partial(archive_check, f, ws, old, 0, digest=False),
        )
    d = balanced(cases, epochs)
    d.update(
        candidate=a.candidate,
        context=a.context,
        left=a.left,
        verdict=block_verdict(d),
        resources={n: chained.resources(ws) for n, ws in spaces.items()},
    )
    (LOG / f"phase410-{a.candidate}-{a.context}-{a.left}.json").write_text(
        json.dumps(d, indent=2) + "\n"
    )
    print(json.dumps(d["timing"]), flush=True)


if __name__ == "__main__":
    main()

"""Run the existing stage suites on independent-resource candidates, log separately."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from layer_fixture import require_gpu0
from layer_reference import exact_check
from reference_archive import checked_load

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_tail,
    independent_tail,
)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--exact-only", action="store_true")
    a = p.parse_args()
    import check_common_s6 as suite

    d = checked_load(f"S6-inputs-{a.seed}.pt")
    exact = []
    for kind in ("normal", "last_weighted", "last_unweighted"):
        pre = None if kind == "last_unweighted" else d["pre"]
        for sign in (1, -1):
            outputs = []
            spaces = []
            for module in (common_tail, independent_tail):
                ws = module.make_workspace("cuda")
                spaces.append(ws)
                outputs.append(
                    module.unpool_boundary(
                        d["x"] * sign,
                        d["weight"],
                        d["stream"],
                        d["post"],
                        pre,
                        d["alpha"],
                        d["beta"],
                        pre_bf16=kind != "normal",
                        workspace=ws,
                    )
                )
            torch.cuda.synchronize()
            for name, aa, bb in [
                ("projection", spaces[0].projection, spaces[1].projection),
                ("out", outputs[0][0], outputs[1][0]),
                ("stream", outputs[0][1], outputs[1][1]),
            ]:
                dtype = torch.int16 if aa.dtype == torch.bfloat16 else torch.int32
                exact_check(
                    aa.view(dtype),
                    bb.view(dtype),
                    f"S6 {kind} sign{sign} {name} bits",
                    {"_rank": 0},
                )
            exact.append(dict(kind=kind, sign=sign, bit_exact=True))
    Path(f"/home/bangtliu/coord/logs/phase49-S6-exact-{a.seed}.json").write_text(
        json.dumps(exact, indent=2) + "\n"
    )
    if a.exact_only:
        return
    original_exercise = suite.exercise

    def exercise(*args):
        return original_exercise(*args[:-1], "49-" + args[-1])

    with (
        patch.object(common_tail, "make_workspace", independent_tail.make_workspace),
        patch.object(common_tail, "unpool_boundary", independent_tail.unpool_boundary),
        patch.object(suite, "exercise", exercise),
    ):
        rows = suite.s6(a.seed)
    Path(f"/home/bangtliu/coord/logs/phase49-S6-suite-{a.seed}.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

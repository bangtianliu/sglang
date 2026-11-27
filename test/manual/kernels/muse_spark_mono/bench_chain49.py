"""Selected independently budgeted chain versus the actual regular layer graph."""

import argparse
import json
from pathlib import Path

from bench_chained47 import setup
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import chained


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    a = p.parse_args()
    cases, native, ws, errors, routes = setup(a.context, a.left)
    d = balanced(cases, {"regular": native.epochs[0], "chained": ws.s3.state.epochs})
    d.update(
        context=a.context,
        left=a.left,
        rank=0,
        seed=7,
        stage_policy=ws.stage_policy,
        regular_errors=errors,
        routing=routes,
        resources=chained.resources(ws),
        launches=6
        + (2 if ws.stage_policy[0] == "wide" else 0)
        + (3 if ws.stage_policy[1] == "k9" else 0)
        + (1 if ws.stage_policy[2] == "wide" else 0),
    )
    d["ratio"] = (
        d["timing"]["chained"]["median_us"] / d["timing"]["regular"]["median_us"]
    )
    Path(
        f"/home/bangtliu/coord/logs/phase49-chain-vs-regular-{a.context}-{a.left}.json"
    ).write_text(json.dumps(d, indent=2) + "\n")
    print(
        json.dumps(
            {
                n: t
                for n, t in d.items()
                if n not in ("raw", "regular_errors", "routing")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

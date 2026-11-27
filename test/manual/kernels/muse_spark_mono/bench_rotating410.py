"""Cold-weight chain/regular comparison: three independently allocated copies."""

import argparse
import json
from pathlib import Path

from bench_chained47 import setup
from ci410 import block_verdict
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import chained


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--prefix", default="phase410-final")
    a = p.parse_args()
    data = [setup(a.context, a.left) for _ in range(3)]
    pointers = {
        key: [v[1].w[key].data_ptr() for v in data]
        for key in ("qkvg", "o_proj", "joint_router_pooler", "unpooler", "w13", "w2")
    }
    assert all(len(set(x)) == 3 for x in pointers.values()), pointers
    cases = {}
    epochs = {}
    for label in ("regular", "chained"):

        def prepare(label=label):
            for d in data:
                d[0][label][0]()

        def call(label=label):
            for d in data:
                d[0][label][1]()

        def check(label=label):
            for d in data:
                d[0][label][2]()

        cases[label] = (prepare, call, check)
        epochs[label] = (
            data[0][1].epochs[0] if label == "regular" else data[0][2].s3.state.epochs
        )
    d = balanced(cases, epochs)
    d.update(
        context=a.context,
        left=a.left,
        weight_copies=3,
        distinct_pointers=pointers,
        unit="microseconds per complete three-copy rotation; per-layer tables divide by 3",
        verdict=block_verdict(d, "regular", "chained"),
        resources=chained.resources(data[0][2]),
        native_errors=[v[3] for v in data],
        routing=[v[4] for v in data],
    )
    path = Path("/home/bangtliu/coord/logs") / f"{a.prefix}-{a.context}-{a.left}.json"
    path.write_text(json.dumps(d, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                per_layer_us={n: v["median_us"] / 3 for n, v in d["timing"].items()},
                verdict=d["verdict"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

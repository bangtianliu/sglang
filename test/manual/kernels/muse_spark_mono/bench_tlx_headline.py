"""Cold dependent mixed TLX chain vs true regular graph and frozen Gluon chain.

Three distinct weight copies; same-process paired orders and epoch parities.
Gluon production sources are checked against the 2d5315ec96 reference before run.
"""

import argparse
import functools
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from bench_chained47 import setup
from check_chained47 import archive_check
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from reference_archive import gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain

ROOT = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--label", default="phaseT4")
    a = p.parse_args()
    changed = subprocess.check_output(
        [
            "git",
            "-c",
            "safe.directory=/home/bangtliu/sglang-mono",
            "diff",
            "--name-only",
            "2d5315ec96..HEAD",
            "--",
            "python/sglang/kernels/plugin_kernels/amd/muse_spark/gfx1250",
        ],
        text=True,
    ).splitlines()
    assert all(Path(x).name.startswith("tlx_") for x in changed), changed
    groups = {n: [] for n in ["regular", "gluon", "mixed"]}
    epochs = {}
    spaces = []
    retained = []
    selected = []
    for _ in range(3):
        base, native, oldws, errors, routes = setup(a.context, a.left)
        retained.append((base, native, oldws, errors, routes))
        groups["regular"].append(base["regular"])
        groups["gluon"].append(base["chained"])
        x, stream, w, m, _ = base["chained"][1].args
        ws = tlx_chain.make_workspace("cuda", context=a.context, left=a.left)
        spaces.append(ws)
        selected.append(ws.tlx_stages)
        old = gpu(
            torch.load(
                Path("/home/bangtliu/coord/phase44-baseline")
                / f"layer-7-{a.context}-{a.left}.pt",
                weights_only=True,
            )["rows"][0]
        )
        prep = functools.partial(tlx_chain.prepare_peers, ws, *native.partials, rank=0)
        call = functools.partial(tlx_chain.launch, x, stream, w, m, ws, rank=0)
        check = functools.partial(archive_check, native.f, ws, old, 0, digest=False)
        groups["mixed"].append((prep, call, check))
        if len(spaces) == 1:
            epochs = dict(
                regular=native.epochs[0],
                gluon=oldws.s3.state.epochs,
                mixed=ws.s3.state.epochs,
            )
    allocations = {}
    for name in (
        "qkvg",
        "o_proj",
        "joint_router_pooler",
        "w13",
        "w13_scale",
        "w2",
        "w2_scale",
        "unpooler",
    ):
        tensors = [entry[0]["chained"][1].args[2][name] for entry in retained]
        allocations[name] = dict(
            distinct=len({t.data_ptr() for t in tensors}),
            bytes_per_copy=tensors[0].numel() * tensors[0].element_size(),
        )
        assert allocations[name]["distinct"] == 3, (name, allocations[name])
    cases = {}
    for label, rows in groups.items():

        def action(index, rows=rows):
            for row in rows:
                row[index]()

        cases[label] = tuple(functools.partial(action, i) for i in range(3))
    pairs = {}
    for ref in ("regular", "gluon"):
        result = balanced(
            {ref: cases[ref], "mixed": cases["mixed"]},
            {ref: epochs[ref], "mixed": epochs["mixed"]},
        )
        result["ratios"] = {
            o: result["timing"]["mixed"]["by_order"][o]["median_us"]
            / result["timing"][ref]["by_order"][o]["median_us"]
            for o in ("AB", "BA")
        }
        result["per_copy_us"] = {
            n: {o: d["by_order"][o]["median_us"] / 3 for o in ("AB", "BA")}
            for n, d in result["timing"].items()
        }
        pairs[ref] = result
        print(
            json.dumps(
                dict(
                    reference=ref,
                    ratios=result["ratios"],
                    per_copy_us=result["per_copy_us"],
                )
            ),
            flush=True,
        )
    result = dict(
        weight_allocations=allocations,
        context=a.context,
        left=a.left,
        copies=3,
        rank=0,
        selected=selected,
        gluon_commit="2d5315ec96",
        gluon_source_changed_paths=changed,
        source_check="all changed files have tlx_ prefix; all Gluon sources unchanged",
        pairs=pairs,
        resources=[tlx_chain.resources(ws) for ws in spaces],
        regular_errors=[x[3] for x in retained],
        routing=[x[4] for x in retained],
        protocol="Three distinct weights; actual dependent regular/mixed/Gluon graphs; prepare excluded; pairwise interleaved same process; both orders and epoch parities balanced; whole-graph event us divided by three for per-layer rows; no stage events",
    )
    path = ROOT / f"{a.label}-headline-{a.context}-{a.left}.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = path.with_name(path.stem + "-" + stamp + path.suffix)
    result["archive"] = str(archive)
    text = json.dumps(result, indent=2, default=str) + "\n"
    archive.write_text(text)
    path.write_text(text)


if __name__ == "__main__":
    main()

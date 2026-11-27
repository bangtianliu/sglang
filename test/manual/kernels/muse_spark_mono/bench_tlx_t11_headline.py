"""T11 cold dependent TLX chain vs true regular graph and frozen Gluon and T10 chains.

Three distinct weight copies; same-process paired orders and epoch parities.
Gluon production sources are checked against the 2d5315ec96 reference before run.
"""

import argparse
import functools
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from bench_chained47 import setup
from check_chained47 import archive_check
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from reference_archive import gpu
from tlx_launch_audit import audited_tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain

ROOT = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--label", default="phaseT11")
    p.add_argument("--t10-reference", type=Path)
    a = p.parse_args()
    t10 = None
    if a.t10_reference is not None:
        manifest = json.loads((a.t10_reference / "manifest.json").read_text())

        def archived(name):
            row = manifest["files"][name]
            path = a.t10_reference / (name + ".py")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            spec = importlib.util.spec_from_file_location(
                "tlx_reference_t10_" + name, path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

        core = Path(tlx_chain.__file__).with_name("tlx_expert_core.py")
        assert (
            hashlib.sha256(core.read_bytes()).hexdigest()
            == manifest["expert_core_sha256"]
        )
        t10 = archived("chain")
        t10.tlx_attn_tail = archived("s3")
        t10.tlx_route = archived("s4")
        t10.tlx_experts = archived("s5")
        t10.tlx_tail = archived("s6")
        t10.tlx_qkvg = archived("s1")
        t10.tlx_attention = archived("s2")
        core = archived("expert_core")
        wide = archived("expert_wide")
        wide.fp4_upcast = core.fp4_upcast
        t10.tlx_experts.expert_tile = core.expert_tile
        t10.tlx_experts.expert_tile_wide = wide.expert_tile_wide
        projection = archived("projection")
        for stage in (t10.tlx_qkvg, t10.tlx_attn_tail, t10.tlx_route, t10.tlx_tail):
            stage.tlx_projection = projection
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
    groups = {n: [] for n in ["regular", "gluon", "tlx"]}
    if t10 is not None:
        groups["t10"] = []
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
        groups["tlx"].append((prep, call, check))
        if t10 is not None:
            refws = t10.make_workspace("cuda", context=a.context, left=a.left)
            groups["t10"].append(
                (
                    functools.partial(
                        t10.prepare_peers, refws, *native.partials, rank=0
                    ),
                    functools.partial(t10.launch, x, stream, w, m, refws, rank=0),
                    functools.partial(
                        archive_check, native.f, refws, old, 0, digest=False
                    ),
                )
            )
        if len(spaces) == 1:
            epochs = dict(
                **({"t10": refws.s3.state.epochs} if t10 is not None else {}),
                regular=native.epochs[0],
                gluon=oldws.s3.state.epochs,
                tlx=ws.s3.state.epochs,
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
    for ref in ("regular", "gluon", "t10") if t10 is not None else ("regular", "gluon"):
        result = balanced(
            {ref: cases[ref], "tlx": cases["tlx"]},
            {ref: epochs[ref], "tlx": epochs["tlx"]},
        )
        result["ratios"] = {
            o: result["timing"]["tlx"]["by_order"][o]["median_us"]
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
        t10_reference_manifest=manifest if t10 is not None else None,
        gluon_source_changed_paths=changed,
        source_check="all changed files have tlx_ prefix; all Gluon sources unchanged",
        pairs=pairs,
        resources=[tlx_chain.resources(ws) for ws in spaces],
        regular_errors=[x[3] for x in retained],
        routing=[x[4] for x in retained],
        protocol="Three distinct weights; actual dependent regular/tlx/Gluon graphs; prepare excluded; pairwise interleaved same process; both orders and epoch parities balanced; whole-graph event us divided by three for per-layer rows; no stage events",
    )
    path = ROOT / f"{a.label}-headline-{a.context}-{a.left}.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = path.with_name(path.stem + "-" + stamp + path.suffix)
    result["archive"] = str(archive)
    text = json.dumps(result, indent=2, default=str) + "\n"
    archive.write_text(text)
    path.write_text(text)


if __name__ == "__main__":
    import sys

    with audited_tlx("headline-" + "-".join(sys.argv[1:]).replace("/", "_")):
        main()

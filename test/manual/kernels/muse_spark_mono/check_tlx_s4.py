"""TLX S4 exact Gluon comparison plus native/FP64/ties/replay contract."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from layer_fixture import require_gpu0
from layer_reference import exact_check
from reference_archive import ROOT as REFS
from reference_archive import checked_load
from tlx_ir import dump
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_route as old,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_route as new

ROOT = Path("/home/bangtliu/coord/logs/T3c-S4-suite")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    ROOT.mkdir(exist_ok=True)
    if a.smoke:
        d = checked_load("phase3d-S4-inputs-7.pt")
        outputs = []
        spaces = []
        for module in (old, new):
            print("BEGIN " + module.__name__, flush=True)
            ws = module.make_workspace("cuda")
            spaces.append(ws)
            outputs.append(
                module.routing(d["x"], d["weight"], d["bias"], d["norm"], workspace=ws)
            )
            torch.cuda.synchronize()
        for name, x, y in [
            ("projection", spaces[0].projection, spaces[1].projection)
        ] + [
            (name, x, y)
            for name, x, y in zip(
                ["scores", "ids", "pooled", "sorted", "experts", "num_post"], *outputs
            )
        ]:
            dtype = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
            exact_check(y.view(dtype), x.view(dtype), "S4 " + name, {"_rank": 0})
        print("PASS S4 exact smoke", flush=True)
        return
    import check_common_s4 as suite

    candidate = new.routing
    retained = []
    seen = set()
    records = []

    def compare(*aa, **kw):
        ws = kw["workspace"]
        if id(ws) in seen:
            return candidate(*aa, **kw)
        retained.append(ws)
        rank = kw.get("rank", 0)
        ref = old.make_workspace("cuda")
        original = old.routing(*aa, **dict(kw, workspace=ref))
        torch.cuda.synchronize()
        actual = candidate(*aa, **kw)
        torch.cuda.synchronize()
        for name, x, y in [("projection", ref.projection, ws.projection)] + [
            (name, x, y)
            for name, x, y in zip(
                ["scores", "ids", "pooled", "sorted", "experts", "num_post"],
                original,
                actual,
            )
        ]:
            dtype = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
            exact_check(y.view(dtype), x.view(dtype), "S4 " + name, {"_rank": rank})
        seen.add(id(ws))
        records.append(
            dict(rank=rank, exact=True, resources=list(ws.state.admitted.values()))
        )
        (ROOT / "exact.json").write_text(json.dumps(records, indent=2) + "\n")
        return actual

    suite.LOG = ROOT
    suite.REFS = REFS
    suite.make_workspace = new.make_workspace
    suite.routing = compare
    with patch.object(sys, "argv", ["suite", "--saved-references"]):
        suite.main()


if __name__ == "__main__":
    seen = set()
    original_metadata = CompiledKernel.launch_metadata

    def metadata(kernel, *aa, **kw):
        result = original_metadata(kernel, *aa, **kw)
        fn = kernel.src.fn.fn
        module = fn.__module__.rsplit(".", 1)[-1]
        phase = {
            "project": "projection",
            "finish_rows": "rows",
            "slots": "slots",
            "_launch": "common",
        }.get(fn.__name__)
        if phase and module in ["tlx_route", "tlx_projection", "common_route"]:
            side = "gluon" if module == "common_route" else "tlx"
            key = (phase, side)
            if key not in seen:
                dump(kernel, "T3c-S4/" + phase, side)
                seen.add(key)
        return result

    with patch.object(CompiledKernel, "launch_metadata", metadata):
        main()

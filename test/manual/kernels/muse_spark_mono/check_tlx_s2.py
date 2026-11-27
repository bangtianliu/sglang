"""S2 exact partial/LSE/output parity then existing native/FP64/tail/replays."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import check_common_s2 as suite
import torch
from layer_reference import exact_check
from tlx_ir import dump
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    split_attention as old,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_attention as new,
)

IR_TAG = "T3-S2"
ROOT = Path("/home/bangtliu/coord/logs/phaseT3-S2-suite")


def main():
    global ROOT, IR_TAG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--label", default="")
    label, remaining = parser.parse_known_args()
    if label.label:
        ROOT = ROOT.with_name(label.label + "-S2-suite")
        IR_TAG = label.label + "-S2"
    ROOT.mkdir(exist_ok=True)
    candidate = new.attention
    retained = []
    seen = set()
    records = []
    active = [None]
    ir_seen = set()

    def compare(*aa, **kw):
        ws = kw["workspace"]
        key = id(ws)
        if key in seen:
            return candidate(*aa, **kw)
        retained.append(ws)
        context = kw["context"]
        left = kw.get("window_left", -1)
        page = kw.get("mapping_page_aligned", False)
        hasmap = kw.get("mapping") is not None
        label = f"{context}-{left}-" + (
            "page" if page else "row" if hasmap else "nomap"
        )
        ref = old.make_workspace("cuda", context=context, window_left=left)
        ref_kw = dict(kw, workspace=ref)
        ref_kw.pop("output", None)
        print(json.dumps(dict(begin="gluon", case=label)), flush=True)
        active[0] = (label, "gluon")
        expected = old.attention(*aa, **ref_kw)
        torch.cuda.synchronize()
        print(json.dumps(dict(begin="tlx", case=label)), flush=True)
        active[0] = (label, "tlx")
        actual = candidate(*aa, **kw)
        torch.cuda.synchronize()
        active[0] = None
        for name, x, y in [
            ("partial", ref.partial, ws.partial),
            ("lse", ref.lse, ws.lse),
            ("out", expected, actual),
        ]:
            dtype = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
            exact_check(
                y.view(dtype), x.view(dtype), "S2 " + label + " " + name, {"_rank": 0}
            )
        seen.add(key)
        records.append(
            dict(case=label, exact=True, resources=list(ws.admitted.values()))
        )
        (ROOT / "exact.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)
        return actual

    original = CompiledKernel.launch_metadata

    def metadata(kernel, *aa, **kw):
        result = original(kernel, *aa, **kw)
        if active[0] is not None:
            case, side = active[0]
            phase = {"partials": "partial", "merge": "merge"}.get(
                kernel.src.fn.fn.__name__
            )
            key = (case, side, phase)
            if phase and key not in ir_seen:
                dump(kernel, IR_TAG + "/" + case + "/" + phase, side)
                ir_seen.add(key)
        return result

    suite.stage = new
    suite.ROOT = ROOT
    suite.suite.LOG = ROOT
    with (
        patch.object(sys, "argv", [sys.argv[0]] + remaining),
        patch.object(new, "attention", compare),
        patch.object(CompiledKernel, "launch_metadata", metadata),
    ):
        suite.main()


if __name__ == "__main__":
    main()

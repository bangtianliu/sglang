"""Exact independent Gluon/TLX S5 plus the common native/FP64 suite."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import check_common_s5 as suite
import torch
from layer_reference import exact_check
from tlx_ir import dump
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    independent_experts as old,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_experts as new,
)

ROOT = Path("/home/bangtliu/coord/logs/T3c-S5-fixed-suite")


def main():
    global ROOT, IR_LABEL
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--label", default="T3c-S5-fixed")
    parser.add_argument("--scale-async", action="store_true")
    parser.add_argument("--fuse-norm-sum", action="store_true")
    flags, remaining = parser.parse_known_args()
    ROOT = Path("/home/bangtliu/coord/logs") / (flags.label + "-suite")
    IR_LABEL = flags.label
    if flags.scale_async or flags.fuse_norm_sum:
        original_make = new.make_workspace

        def make_workspace(device):
            ws = original_make(device)
            ws.scale_async = flags.scale_async
            ws.fuse_norm_sum = flags.fuse_norm_sum
            return ws

        new.make_workspace = make_workspace
    ROOT.mkdir(exist_ok=True)
    candidate = new.s5
    retained = []
    seen = set()
    records = []

    def compare(*aa, **kw):
        ws = aa[6]
        if id(ws) in seen:
            return candidate(*aa, **kw)
        retained.append(ws)
        rank = kw["rank"]
        ref = old.make_workspace("cuda")
        ref.inter.fill_(float("nan"))
        ref.output.fill_(float("nan"))
        ref.peers.scratch.copy_(ws.peers.scratch)
        ref.state.epochs.copy_(ws.state.epochs)
        old.s5(*aa[:6], ref, **kw)
        torch.cuda.synchronize()
        k = candidate(*aa, **kw)
        torch.cuda.synchronize()
        valid = (aa[1].flatten() >= rank * 64) & (aa[1].flatten() < (rank + 1) * 64)
        pairs = [
            (name, getattr(ref, name)[rank, valid], getattr(ws, name)[rank, valid])
            for name in ["inter", "output", "scaled"]
        ]
        pairs += [("inv", ref.inv[rank, valid, 0], ws.inv[rank, valid, 0])]
        pairs += [
            (name, getattr(ref, name)[rank], getattr(ws, name)[rank])
            for name in ["weighted", "partial"]
        ]
        pairs += [
            ("reduced", ref.peers.reduced[rank], ws.peers.reduced[rank]),
            ("layer_out", ref.layer_out, ws.layer_out),
        ]
        for name, x, y in pairs:
            dtype = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
            try:
                exact_check(y.view(dtype), x.view(dtype), "S5 " + name, {"_rank": rank})
            except AssertionError:
                torch.save(
                    dict(actual=y.cpu(), expected=x.cpu(), rank=rank, tensor=name),
                    ROOT / f"failure-{name}-r{rank}.pt",
                )
                raise
        seen.add(id(ws))
        records.append(
            dict(rank=rank, exact=True, resources=list(ws.state.admitted.values()))
        )
        (ROOT / "exact.json").write_text(json.dumps(records, indent=2) + "\n")
        return k

    new.s5 = compare
    suite.stage = new
    suite.ROOT = ROOT
    with patch.object(sys, "argv", ["suite", *remaining]):
        suite.main()


if __name__ == "__main__":
    seen = set()
    original = CompiledKernel.launch_metadata

    def metadata(kernel, *aa, **kw):
        out = original(kernel, *aa, **kw)
        fn = kernel.src.fn.fn
        mod = fn.__module__.rsplit(".", 1)[-1]
        name = fn.__name__
        if mod in ["independent_experts", "tlx_experts"]:
            phase = name
            if name in ["_expert", "expert_phase"]:
                idx = kernel.src.fn.arg_names.index("FIRST")
                first = kernel.src.constants.get(
                    (idx,), kernel.src.constants.get("FIRST")
                )
                phase = "w13" if first else "w2"
            side = "tlx" if mod == "tlx_experts" else "gluon"
            key = (side, phase)
            if key not in seen:
                dump(kernel, IR_LABEL + "/" + phase, side)
                seen.add(key)
        return out

    with patch.object(CompiledKernel, "launch_metadata", metadata):
        main()

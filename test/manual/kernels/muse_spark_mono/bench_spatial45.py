"""One bounded spatial split; accepted default and mathematical helpers unchanged."""

import argparse
import functools
import json
from unittest.mock import patch

import spatial45
import torch
from layer_fixture import require_gpu0
from mono_case45 import LOG, case, verify
from paired45 import measure

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.code_probe import (
    text_info,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer


def launch(*args, **kwargs):
    # Host-only selection scoped to this manual harness; no concurrent host calls.
    kwargs.pop("code_warm", None)
    with patch.object(layer.layer_accepted, "_launch", spatial45._launch):
        return layer.launch(*args, **kwargs)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    a = p.parse_args()
    f, w, m, old = case(a.context, a.left)
    cases = {}
    resources = {}
    # The host passes CodeWarm even when None; accept this no-op argument in the split.
    for label in ("temporal", "spatial"):
        ws = layer.make_workspace("cuda", context=a.context, left=a.left)
        call = layer.launch
        if label == "spatial":
            ws.s5.peers = standin.make_standins("cuda", width=4096, ctas=256)
            ws.s5.peers.ready = torch.empty((1,), device="cuda", dtype=torch.int32)
            call = launch
        compiled = call(f.x, f.stream, w, m, ws, rank=0, compile_only=True)
        ws.shape.admit(label, compiled, grid=768)
        resources[label] = dict(
            text_info(compiled)[0],
            vgpr=compiled.n_regs,
            spills=compiled.n_spills,
            lds=compiled.metadata.shared,
        )
        print(
            json.dumps(dict(event="compiled", variant=label, **resources[label])),
            flush=True,
        )
        cases[label] = (
            functools.partial(
                layer.prepare_peers, ws, old["partial3"], old["partial5"], rank=0
            ),
            functools.partial(call, f.x, f.stream, w, m, ws, rank=0),
            functools.partial(verify, f, ws, old),
        )
    result, _ = measure(cases)
    result.update(
        resources=resources, N=512, late_ctas=256, context=a.context, left=a.left
    )
    (LOG / f"phase45-spatial-{a.context}-{a.left}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: {x: y for x, y in v.items() if x != "samples_by_order_us"}
                if isinstance(v, dict)
                else v
                for k, v in result.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

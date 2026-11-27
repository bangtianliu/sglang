"""Mono-only fixtures/checks: deliberately no uTLX or native/runtime imports."""

import sys
from pathlib import Path

import torch
from layer_case import make
from reference_archive import gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer

LOG = Path("/home/bangtliu/coord/logs")


def case(context=8192, left=-1):
    f, w, m = make(7, context, left, 0)
    old = gpu(
        torch.load(
            f"/home/bangtliu/coord/phase44-baseline/layer-7-{context}-{left}.pt",
            weights_only=True,
        )["rows"][0]
    )
    return f, w, m, old


def verify(f, ws, old):
    assert bool(ws.status[:6].eq(1).all()), layer.diagnostic_records(ws)
    assert torch.equal(ws.s4.ids, old["ids"]), "ids"
    for n, e in old["buffers"].items():
        a = ws.buffers[n]
        typ = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
        assert torch.equal(a.view(typ), e.view(typ)), n
    assert torch.equal(
        ws.s4.scores.view(torch.int32), old["scores"].view(torch.int32)
    ), "scores"
    assert torch.equal(ws.s4.pooled.view(torch.int16), old["pool"].view(torch.int16)), (
        "pool"
    )
    assert not any(
        n == "utlx_plugin" or n.startswith("utlx_plugin.") for n in sys.modules
    ), "unexpected uTLX import"

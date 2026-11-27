"""Isolate S4 on the integrated layer's exact X4 before changing layer checks."""

import json
from pathlib import Path

import torch
from check_route import native, oracle
from check_tail import check
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import build

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_route,
    layer,
)


def biased(x, w):
    p = x.double() @ w["joint_router_pooler"].double().T
    raw = ((p[:, 4096:4352] + p[:, 4352:]) * 32**-0.5).sigmoid()
    return raw + w["bias"].double()


def main():
    require_gpu0()
    f, w, m = make(7, 8192, -1, 1)
    ref = build(f, w, 7, 8192, -1)
    ws = layer.make_workspace("cuda", context=8192, left=-1)
    layer.prepare_peers(ws, ref["partial3"], ref["partial5"], rank=1)
    layer.launch(f.x, f.stream, w, m, ws, rank=1)
    torch.cuda.synchronize()
    assert ws.status[:6].eq(1).all()
    x = ws.buffers["x4"]
    stage_ws = common_route.make_workspace("cuda")
    actual = common_route.routing(
        x,
        w["joint_router_pooler"],
        w["bias"],
        w["route_norm"],
        rank=1,
        workspace=stage_ws,
    )
    expected = native(x, w["joint_router_pooler"], w["bias"], w["route_norm"], 1)
    fp64 = oracle(x, w["joint_router_pooler"], w["bias"], w["route_norm"])
    torch.cuda.synchronize()
    candidate = biased(x, w)
    reference = biased(ref["x4"], w)
    top = reference.argsort(descending=True, stable=True)[:, :9]
    row = dict(
        layer_ids=ws.s4.ids.cpu().tolist(),
        standalone_ids=actual[1].cpu().tolist(),
        native_ids=expected[1].cpu().tolist(),
        fp64_ids=fp64[1].cpu().tolist(),
        max_biased_score_perturbation=float((candidate - reference).abs().max()),
        rows=[],
    )
    for r in range(8):
        ri = top[r]
        rr = reference[r, ri]
        cr = candidate[r, ri]
        row["rows"].append(
            dict(
                row=r,
                reference_top9=ri.cpu().tolist(),
                reference_biased_scores=rr.cpu().tolist(),
                candidate_biased_scores_at_reference_ids=cr.cpu().tolist(),
                reference_margins_5_6_7_8_9=(rr[4:8] - rr[5:9]).cpu().tolist(),
                candidate_selected_scores=actual[0][r].cpu().tolist(),
                native_selected_scores=expected[0][r].cpu().tolist(),
                fp64_selected_scores=fp64[0][r].cpu().tolist(),
            )
        )
    Path("/home/bangtliu/coord/logs/phase42-route-isolation.json").write_text(
        json.dumps(row, indent=2) + "\n"
    )
    print(json.dumps(row), flush=True)
    assert torch.equal(actual[1], expected[1]), "S4 identical-input native ID mismatch"
    assert torch.equal(ws.s4.ids, actual[1]), "layer-vs-standalone S4 mismatch"
    assert torch.equal(actual[1], fp64[1]), "S4 identical-input FP64 ID mismatch"
    for i in (0, 2):
        check(actual[i], expected[i])
        check(actual[i], fp64[i])
    print(
        "PASS identical-input S4 exact order/native/FP64; upstream-rounding isolation",
        flush=True,
    )


if __name__ == "__main__":
    main()

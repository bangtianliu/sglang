"""Finite opt3 catalog; every candidate checks FP64 before and after timing."""

import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from layer_fixture import require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attention import (
    attention,
    make_workspace,
)
from sglang.test.kernels.tlx_attention import paged_reference


def main():
    require_gpu0()
    log = Path("/home/bangtliu/coord/logs")
    data = {
        k: v.to("cuda")
        for k, v in torch.load(
            log / "phase3c-S2-opt2-shared-inputs.pt", weights_only=True
        ).items()
    }
    ref = paged_reference(
        dict(**data, softmax_scale=0.125, causal=True, window_size=(2047, 0)),
        [8],
        [131072],
    )
    results = []
    for splits in (0, 4, 8):
        ws = make_workspace(
            data["q"].device, context=131072, window_left=2047, sliding_splits=splits
        )
        call = lambda: attention(
            data["q"],
            data["k"],
            data["v"],
            data["block_table"],
            data["seqused_k"],
            data["k_descale"],
            data["v_descale"],
            context=131072,
            window_left=2047,
            workspace=ws,
            sliding_splits=splits,
        )
        y = call()
        torch.cuda.synchronize()
        rel = ((y.double() - ref).abs().max() / ref.abs().max()).item()
        if not torch.isfinite(y).all() or rel >= 0.008 or not ws.status.eq(1).all():
            result = dict(
                splits=splits, rejected=True, rel=rel, status=ws.status.tolist()
            )
        else:
            timing = graph_times(call)
            y = call()
            torch.cuda.synchronize()
            assert (
                ws.status.eq(1).all()
                and ((y.double() - ref).abs().max() / ref.abs().max()).item() < 0.008
            )
            result = dict(splits=splits, rel=rel, timing=timing, rejected=False)
        print(json.dumps(result), flush=True)
        results.append(result)
        (log / "phase3c-S2-opt3-window-sweep.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()

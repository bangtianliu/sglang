"""Instrumentation only: calibrated per-CTA S4/S6 boundaries, private trace lines."""

import json
from pathlib import Path

import numpy as np
import torch
from bench_hbm import graph_times
from layer_fixture import require_gpu0
from profile_s2 import calibrate

LOG = Path("/home/bangtliu/coord/logs")


def run_stage(name, rate):
    if name == "S4":
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
            route as module,
        )

        d = {
            k: v.cuda()
            for k, v in torch.load(
                LOG / "phase3d-S4-inputs-7.pt", weights_only=True
            ).items()
        }
        ws = module.make_workspace(d["x"].device)
        s = ws.state
        args = (
            d["x"],
            d["weight"],
            d["bias"],
            d["norm"],
            ws.projection,
            s.epochs,
            s.arrivals,
            s.flags,
            s.status,
            ws.scores,
            ws.ids,
            ws.pooled,
            ws.sorted_ids,
            ws.expert_ids,
            ws.num_post,
            0,
        )
        outputs = (
            ws.scores,
            ws.ids,
            ws.pooled,
            ws.sorted_ids,
            ws.expert_ids,
            ws.num_post,
        )
        active = 144
        phases = [
            (0, 12, "epoch_and_setup", 192),
            (12, 1, "active_gemm_and_publish", active),
            (1, 2, "first_handoff_active", active),
            (1, 2, "first_handoff_row_consumers", 8),
            (2, 3, "router_loads_sigmoid_bias", 8),
            (3, 4, "top8_and_stores", 8),
            (4, 5, "pooler_loads_round", 8),
            (5, 6, "pooler_norm_store", 8),
            (6, 7, "second_handoff_row_consumers", 8),
            (7, 8, "slot_ids_counts_offsets", 1),
            (8, 9, "serial_slot_loop", 1),
            (9, 10, "slot_block_metadata", 1),
            (10, 11, "status_after_slots", 1),
        ]
    else:
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
            tail as module,
        )

        d = {
            k: v.cuda()
            for k, v in torch.load(
                LOG / "phase3d-S6-inputs-7.pt", weights_only=True
            ).items()
        }
        ws = module.make_workspace(d["x"].device)
        s = ws.state
        out = torch.empty((8, 8192), device="cuda", dtype=torch.bfloat16)
        stream = torch.empty((8, 8192), device="cuda")
        args = (
            d["x"],
            d["weight"],
            d["stream"],
            d["post"],
            d["pre"],
            d["alpha"],
            d["beta"],
            ws.projection,
            s.epochs,
            s.arrivals,
            s.flags,
            s.status,
            out,
            stream,
            False,
        )
        outputs = (out, stream)
        active = 128
        phases = [
            (0, 12, "epoch_and_setup", 192),
            (12, 1, "active_gemm_and_publish", active),
            (1, 2, "first_handoff_active", active),
            (1, 2, "first_handoff_row_consumers", 8),
            (2, 3, "branch_loads_round", 8),
            (3, 4, "post_input_scale_round", 8),
            (4, 5, "post_norm_round", 8),
            (5, 6, "gated_stream_update_store", 8),
            (6, 7, "pre_norm", 8),
            (7, 8, "output_store", 8),
            (8, 11, "status_after_output", 8),
        ]

    def run(trace=None):
        return module._standalone[(192,)](
            *args, TRACE=trace, num_warps=16, enable_fp_fusion=False
        )

    kernel = run()
    torch.cuda.synchronize()
    assert s.status.eq(1).all()
    expected = tuple(t.clone() for t in outputs)
    uninstrumented = graph_times(run)
    trace = torch.zeros((192, 16), device="cuda", dtype=torch.uint64)
    samples = []
    for _ in range(10):
        trace.zero_()
        kernel_trace = run(trace)
        torch.cuda.synchronize()
        assert s.status.eq(1).all()
        assert all(torch.equal(a, b) for a, b in zip(outputs, expected)), name
        samples.append(trace.cpu().numpy().copy())
    raw = np.stack(samples)
    np.save(LOG / f"phase3e0-{name}-trace.npy", raw)
    (LOG / f"phase3e0-{name}-instrumented.amdgcn").write_text(
        kernel_trace.asm["amdgcn"]
    )
    (LOG / f"phase3e0-{name}-uninstrumented.amdgcn").write_text(kernel.asm["amdgcn"])
    result = dict(
        stage=name,
        ticks_per_us=rate,
        samples=10,
        uninstrumented=uninstrumented,
        trace_outputs_bitwise=True,
        phases_us={},
    )
    for a, b, label, count in phases:
        values = (
            raw[:, :count, b].astype(np.int64) - raw[:, :count, a].astype(np.int64)
        ) / rate
        valid = (raw[:, :count, a] > 0) & (raw[:, :count, b] > 0)
        selected = values[valid]
        assert selected.size > 0, (name, label)
        result["phases_us"][label] = dict(
            ctas=count,
            median=float(np.median(selected)),
            p99=float(np.quantile(selected, 0.99)),
            max=float(selected.max()),
        )
    print(json.dumps(result), flush=True)
    return result


def main():
    require_gpu0()
    rate = calibrate()
    results = [run_stage(name, rate) for name in ("S4", "S6")]
    (LOG / "phase3e0-breakdown.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

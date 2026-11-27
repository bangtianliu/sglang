"""Calibrated realtime-CTA phase boundaries; instrumentation excluded from latency."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from bench_hbm import graph_times
from layer_fixture import build_layer, require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync


@g.jit
def mark(OUT, OFFSET: gl.constexpr):
    gl.store(OUT + OFFSET, sync.clock())


def calibrate():
    t = torch.empty(2, device="cuda", dtype=torch.uint64)
    mark[(1,)](t, 0, num_warps=4)
    torch.cuda.synchronize()
    rates = []
    for _ in range(3):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        mark[(1,)](t, 0, num_warps=4)
        torch.cuda.synchronize()
        time.sleep(0.02)
        mark[(1,)](t, 1, num_warps=4)
        b.record()
        torch.cuda.synchronize()
        ticks = t.cpu().tolist()
        rates.append((ticks[1] - ticks[0]) / (a.elapsed_time(b) * 1000))
    return float(np.median(rates))


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--before", action="store_true")
    p.add_argument("--bn", type=int, default=64)
    p.add_argument("--waves", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=2)
    args = p.parse_args()
    if args.before:
        spec = importlib.util.spec_from_file_location(
            "s2_before_profile",
            "/home/bangtliu/coord/logs/phase3c-S2-opt-before-profile.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
            attention as module,
        )
    bn = 128 if args.before else args.bn
    waves = 8 if args.before else args.waves
    rate = calibrate()
    results = []
    for left in (-1, 2047):
        f = build_layer(context=131072, components=())
        q = torch.randn((8, 32, 64), device="cuda", dtype=torch.bfloat16)
        ws = module.make_workspace(
            q.device, context=131072, window_left=left, block_n=bn
        )
        out = torch.empty_like(q)
        trace = torch.zeros((256, 8), device="cuda", dtype=torch.uint64)

        def run(profile):
            return module._standalone[(192,)](
                q,
                f.cache_k,
                f.cache_v,
                f.page_table,
                None,
                f.seq_lens,
                f.k_scale,
                f.v_scale,
                ws.partial,
                ws.lse,
                ws.counters,
                ws.flags,
                ws.status,
                out,
                0.125,
                left,
                ws.splits,
                bn,
                False,
                **(
                    {}
                    if args.before
                    else dict(MAX_PAGES=ws.pages_per_chunk, PREFETCH=args.prefetch)
                ),
                TRACE=trace if profile else None,
                num_warps=waves,
                enable_fp_fusion=False,
            )

        kernel = run(False)
        Path(
            "/home/bangtliu/coord/logs/phase3c-S2-opt-"
            + ("before" if args.before else "after")
            + f"-left{left}.amdgcn"
        ).write_text(kernel.asm["amdgcn"])
        timing = graph_times(lambda: run(False))
        samples = []
        for _ in range(10):
            run(True)
            torch.cuda.synchronize()
            assert ws.status.eq(1).all()
            samples.append(trace.cpu().numpy().copy())
        values = np.stack(samples)
        item = dict(left=left, ticks_per_us=rate, uninstrumented=timing, phases_us={})
        for i, name in enumerate(
            ("partial", "publish", "flag_wait", "merge_loads", "merge_math", "output")
        ):
            delta = (
                values[:, :, i + 1].astype(np.int64) - values[:, :, i].astype(np.int64)
            ) / rate
            valid = (values[:, :, i] != 0) & (values[:, :, i + 1] != 0)
            v = delta[valid]
            item["phases_us"][name] = dict(
                median=float(np.median(v)),
                p99=float(np.quantile(v, 0.99)),
                max=float(v.max()),
            )
        if not args.before:
            for a, b, name in [
                (3, 7, "merge_raw_b128"),
                (7, 4, "merge_layout_and_lse"),
            ]:
                delta = (
                    values[:, :, b].astype(np.int64) - values[:, :, a].astype(np.int64)
                ) / rate
                v = delta[(values[:, :, b] != 0) & (values[:, :, a] != 0)]
                item["phases_us"][name] = dict(
                    median=float(np.median(v)),
                    p99=float(np.quantile(v, 0.99)),
                    max=float(v.max()),
                )
        results.append(item)
        print(json.dumps(item), flush=True)
    Path(
        "/home/bangtliu/coord/logs/phase3c-S2-opt-"
        + ("before" if args.before else "after")
        + "-breakdown.json"
    ).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

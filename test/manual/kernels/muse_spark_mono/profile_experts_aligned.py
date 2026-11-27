"""Matched original/aligned FP4 WMMA loop slopes, with resident constant A.

This is a loop proxy, not an HBM bandwidth floor: each CTA cycles its tile's
16 K packets. Timing fits remove fixed clock and output-reduction overhead.
"""

import json
from pathlib import Path

import numpy as np
import torch
from bench_hbm import graph_times
from check_experts import load
from layer_fixture import require_gpu0
from profile_s2 import calibrate
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    experts_aligned as aligned,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    experts_routed as old,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts import (
    _fp4_upcast,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import _mma

ROOT = Path("/home/bangtliu/coord/logs")


@g.jit
def loop(W, Scale, Sink, Trace, ITER: gl.constexpr, ALIGNED: gl.constexpr):
    mma: gl.constexpr = _mma(16, 64)
    a = gl.full((4, 16, 64), 0.001, gl.bfloat16, gl.DotOperandLayout(0, mma, 8))
    acc = gl.full((4, 16, 64), 0, gl.float32, mma)
    expert = gl.program_id(0) % 19
    tile = gl.program_id(0) // 19
    p0, s0 = (aligned.weight_packets if ALIGNED else old._weights)(
        W, Scale, expert, tile, 0, 8192, 64, 64, 4
    )
    p1, s1 = (aligned.weight_packets if ALIGNED else old._weights)(
        W, Scale, expert, tile, 1, 8192, 64, 64, 4
    )
    p2, s2 = (aligned.weight_packets if ALIGNED else old._weights)(
        W, Scale, expert, tile, 2, 8192, 64, 64, 4
    )
    rt0 = sync.clock()
    cy0 = sync.shader_cycles()
    for step in range(ITER):
        if ALIGNED:
            b = aligned.upcast(p0, s0)
        else:
            b = _fp4_upcast(p0, s0, 64, 64, 4, "direct")
        acc = gfx1250.wmma(a, b, acc)
        p0 = p1
        s0 = s1
        p1 = p2
        s1 = s2
        p2, s2 = (aligned.weight_packets if ALIGNED else old._weights)(
            W, Scale, expert, tile, (step + 3) % 16, 8192, 64, 64, 4
        )
    gl.store(Sink + gl.program_id(0), gl.sum(gl.sum(gl.sum(acc, 0), 0), 0))
    cy1 = sync.shader_cycles()
    rt1 = sync.clock()
    gl.store(Trace + gl.program_id(0) * 4, cy0)
    gl.store(Trace + gl.program_id(0) * 4 + 1, cy1)
    gl.store(Trace + gl.program_id(0) * 4 + 2, rt0)
    gl.store(Trace + gl.program_id(0) * 4 + 3, rt1)


def main():
    require_gpu0()
    rate = calibrate()
    x, w = load(7)
    trace = torch.empty((192, 4), device=x.device, dtype=torch.uint64)
    sink = torch.empty(192, device=x.device)
    rows = []
    fits = []
    for direct in (False, True):
        group = []
        for iterations in (64, 256, 1024):
            call = lambda: loop[(192,)](
                w["w13"],
                w["w13_scale"],
                sink,
                trace,
                iterations,
                direct,
                num_warps=16,
                enable_fp_fusion=False,
            )
            k = call()
            torch.cuda.synchronize()
            v = trace.cpu().to(torch.int64)
            assert sink.isfinite().all()
            r = dict(
                aligned=direct,
                iterations=iterations,
                cycles=float((v[:, 1] - v[:, 0]).double().median()),
                realtime=float((v[:, 3] - v[:, 2]).double().median()),
                registers=k.n_regs,
                spills=k.n_spills,
                shared=k.metadata.shared,
                **graph_times(call, samples=30),
            )
            if iterations == 1024:
                (ROOT / f"phase3e1d-loop-aligned{int(direct)}.amdgcn").write_text(
                    k.asm["amdgcn"]
                )
            print(json.dumps(r), flush=True)
            group.append(r)
            rows.append(r)
        slopes = {
            name: float(
                np.polyfit(
                    [r["iterations"] for r in group], [r[name] for r in group], 1
                )[0]
            )
            for name in ("cycles", "realtime", "median_us")
        }
        fit = dict(
            aligned=direct,
            cycles_per_iteration=slopes["cycles"],
            realtime_us_per_iteration=slopes["realtime"] / rate,
            event_us_per_iteration=slopes["median_us"],
        )
        print(json.dumps(fit), flush=True)
        fits.append(fit)
    (ROOT / "phase3e1d-loop.json").write_text(
        json.dumps(dict(ticks_per_us=rate, rows=rows, fits=fits), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

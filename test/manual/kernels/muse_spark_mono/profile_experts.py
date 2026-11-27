"""S5 iteration ablations: load/conversion/math proxies, not additive components."""

import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from layer_fixture import build_layer, require_gpu0
from profile_s2 import calibrate
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts import (
    _fp4_gemm,
    _fp4_raw,
    _fp4_upcast,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import (
    _input,
    _mma,
    load_input,
)


@g.jit
def probe(X, W, Scale, Out, Trace, OP: gl.constexpr, MODE: gl.constexpr):
    tile = gl.program_id(0)
    mma: gl.constexpr = _mma(16, 64)
    cached = load_input(X, 4096, 4, 64)
    if OP == "conversion" or OP == "math":
        p, s = _fp4_raw(W, Scale, 3, tile, 0, 4096, 64, 64, 4)
    if OP == "math":
        a = _input(cached, 0, 4096, 64, 4, mma)
        b = _fp4_upcast(p, s, 64, 64, 4, MODE)
    start = sync.clock()
    if OP == "full":
        v = _fp4_gemm(W, Scale, cached, 3, tile, 4096, 64, 64, MODE)
        checksum = gl.sum(gl.sum(v, 0), 0)
    elif OP == "math":
        acc = gl.full((4, 16, 64), 0, gl.float32, mma)
        for step in range(16):
            acc = gfx1250.wmma(a, b, acc)
        checksum = gl.sum(gl.sum(gl.sum(acc, 0), 0), 0)
    elif OP == "loads":
        raw_layout: gl.constexpr = gl.BlockedLayout(
            [1, 16, 1], [1, 8, 4], [4, 1, 4], [1, 2, 0]
        )
        accum = gl.full((4, 32, 64), 0, gl.float32, raw_layout)
        for step in range(16):
            packed, scale = _fp4_raw(W, Scale, 3, tile, step, 4096, 64, 64, 4)
            accum += packed.to(gl.float32) + scale.to(gl.float32)
        checksum = gl.sum(gl.sum(gl.sum(accum, 0), 0), 0)
    else:
        accum = gl.full((4, 64, 64), 0, gl.float32, gl.DotOperandLayout(1, mma, 8))
        for step in range(16):
            term = _fp4_upcast((p ^ step).to(gl.uint8), s, 64, 64, 4, MODE).to(
                gl.float32
            )
            accum += term
        checksum = gl.sum(gl.sum(gl.sum(accum, 0), 0), 0)
    gl.store(Out + tile, checksum)
    end = sync.clock()
    gl.store(Trace + tile * 2, start)
    gl.store(Trace + tile * 2 + 1, end)


def main():
    require_gpu0()
    f = build_layer(seed=7, components=("experts",))
    w = f.weights
    x = f.x[:, :4096].contiguous()
    out = torch.empty(64, device="cuda")
    trace = torch.empty((64, 2), device="cuda", dtype=torch.uint64)
    rate = calibrate()
    records = []
    print("clock_rate", rate, flush=True)
    for mode in ("direct", "alu"):
        for op in ("loads", "conversion", "math", "full"):
            call = lambda: probe[(64,)](
                x,
                w["w2"],
                w["w2_scale"],
                out,
                trace,
                op,
                mode,
                num_warps=16,
                enable_fp_fusion=False,
            )
            k = call()
            torch.cuda.synchronize()
            t = trace.cpu().to(torch.int64)
            dt = (t[:, 1] - t[:, 0]).double() / rate
            row = dict(
                mode=mode,
                operation=op,
                cta_median_us=dt.median().item(),
                cta_max_us=dt.max().item(),
                registers=k.n_regs,
                spills=k.n_spills,
                **graph_times(call),
            )
            print(json.dumps(row), flush=True)
            records.append(row)
            Path(
                f"/home/bangtliu/coord/logs/phase3e1-ablation-{mode}-{op}.amdgcn"
            ).write_text(k.asm["amdgcn"])
    Path("/home/bangtliu/coord/logs/phase3e1-ablations.json").write_text(
        json.dumps(dict(ticks_per_us=rate, records=records), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

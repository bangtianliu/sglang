"""Isolate S5 WMMA/dequant core from routing and handoffs."""

import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from layer_fixture import build_layer, require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts import fp4_tile
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import (
    _mma,
    load_input,
)


@g.jit
def core(X, W, S, Out, PAIR: gl.constexpr, MODE: gl.constexpr):
    tile = gl.program_id(0)
    cached = load_input(X, 4096, 4, 64)
    v = fp4_tile(W, S, cached, 3, tile, 8192 if PAIR else 4096, PAIR=PAIR, MODE=MODE)
    layout: gl.constexpr = gl.SliceLayout(0, _mma(16, 64))
    r = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    n = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    gl.store(Out + r[:, None] * 4096 + tile * 64 + n[None, :], v, r[:, None] < 8)


def main():
    require_gpu0()
    f = build_layer(seed=7, components=("experts",))
    w = f.weights
    x = f.x[:, :4096].contiguous()
    out = torch.empty((8, 4096), device="cuda", dtype=torch.bfloat16)
    for pair in (True, False):
        name = "w13" if pair else "w2"
        for mode in ("direct", "unit", "alu"):
            call = lambda: core[(64,)](
                x,
                w[name],
                w[name + "_scale"],
                out,
                pair,
                mode,
                num_warps=16,
                enable_fp_fusion=False,
            )
            k = call()
            torch.cuda.synchronize()
            record = dict(
                name=name,
                mode=mode,
                registers=k.n_regs,
                spills=k.n_spills,
                shared=k.metadata.shared,
                **graph_times(call),
            )
            print(json.dumps(record), flush=True)
            Path(
                f"/home/bangtliu/coord/logs/phase3e1-core-{name}-{mode}.amdgcn"
            ).write_text(k.asm["amdgcn"])


if __name__ == "__main__":
    main()

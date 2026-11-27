"""Bounded native-layout MXFP4 converter inventory; GPU0 only."""

import json
from pathlib import Path

import torch
from layer_fixture import require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts import (
    _fp4_raw,
    _fp4_upcast,
)
from sglang.srt.rocm_muse_spark.muse_moe_triton import dequant_mxfp4


@g.jit
def convert(W, Scale, Out, MODE: gl.constexpr):
    packed, scale = _fp4_raw(W, Scale, 0, 0, 0, 32, 32, 128, 8, 1024)
    value = _fp4_upcast(packed, scale, 32, 128, 8, MODE)
    layout: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 8, 4], [8, 1, 2], [1, 2, 0])
    value = gl.convert_layout(value, layout)
    s = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    n = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    gl.store(
        Out + n[None, None, :] * 1024 + s[:, None, None] * 128 + k[None, :, None], value
    )


def main():
    require_gpu0()
    torch.manual_seed(7)
    packed = torch.randint(0, 256, (1, 32, 512), device="cuda", dtype=torch.uint8)
    scale = torch.randint(116, 124, (1, 32, 32), device="cuda", dtype=torch.uint8)
    reference = dequant_mxfp4(packed, scale).bfloat16()
    rows = []
    for mode in ("alu", "direct", "unit"):
        output = torch.empty((1, 32, 1024), device="cuda", dtype=torch.bfloat16)
        compiled = convert[(1,)](
            packed, scale, output, mode, num_warps=16, enable_fp_fusion=False
        )
        torch.cuda.synchronize()
        mismatch = output.ne(reference).sum().item()
        row = dict(
            mode=mode,
            mismatches=mismatch,
            max_abs=(output.float() - reference.float()).abs().max().item(),
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
        Path(f"/home/bangtliu/coord/logs/phase3e1-fp4-{mode}.amdgcn").write_text(
            compiled.asm["amdgcn"]
        )
    assert rows[0]["mismatches"] == rows[2]["mismatches"] == 0
    Path("/home/bangtliu/coord/logs/phase3e1-fp4-converter.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

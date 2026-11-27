"""CPU-only exact-target S6 boundary IR for local pin placement checks."""

import argparse
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_tail import finish

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
signature = dict(
    P="*i32",
    Stream="*fp32",
    Post="*bf16",
    Pre="*bf16",
    Alpha="*fp32",
    Beta="*fp32",
    Out="*bf16",
    Updated="*fp32",
    Status="*i32",
    Failure="*i32",
)
src = ASTSource(
    finish,
    signature=signature,
    constexprs=dict(PRE_BF16=False, PIN_LAYOUT=True),
    attrs={
        (i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(10)
    },
)
k = triton.compile(
    src,
    target=GPUTarget("hip", "gfx1250", 32),
    options=dict(num_warps=4, enable_fp_fusion=False),
)
for level in ("ttir", "ttgir", "llir", "amdgcn"):
    (a.out / ("tlx." + level)).write_text(k.asm[level])

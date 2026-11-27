"""CPU-only gfx1250 compile of TLX S1 before any GPU launch."""

from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_projection,
    tlx_qkvg,
)

root = Path("/home/bangtliu/coord/logs/phaseT-ir/T3-S1-compile")
root.mkdir(exist_ok=True)
for name, fn, sig, const, warps in [
    (
        "heads",
        tlx_qkvg.qkvg_heads,
        dict(
            P="*i32",
            State="*i32",
            Status="*i32",
            POS="*i64",
            COS="*bf16",
            LOC="*i64",
            MAP="*i64",
            KC="*fp8e4nv",
            VC="*fp8e4nv",
            Q="*bf16",
            K="*bf16",
            V="*bf16",
            G="*bf16",
            KS="*fp32",
            VS="*fp32",
            Failure="*i32",
        ),
        dict(EPS=1e-5, SCALE=5.4730064863875, ROPE=True, HAS_MAP=True, Seed=None),
        4,
    ),
    (
        "projection",
        tlx_projection.project,
        dict(X="*bf16", W="*bf16", P="*i32", Status="*i32", Failure="*i32"),
        dict(
            N=4608,
            K=8192,
            STAGE=1,
            SK=8,
            PINNED=True,
            POST_BARRIER=True,
            ROUND_OUTPUT=True,
            SKIP=-1,
            Seed=None,
            Packed=None,
        ),
        2,
    ),
]:
    src = ASTSource(
        fn,
        signature=sig,
        constexprs=const,
        attrs={
            (i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
            for i in range(len(sig))
        },
    )
    k = triton.compile(
        src,
        target=GPUTarget("hip", "gfx1250", 32),
        options=dict(num_warps=warps, enable_fp_fusion=False),
    )
    folder = root / name
    folder.mkdir(exist_ok=True)
    for ext in ("ttgir", "llir", "amdgcn"):
        (folder / f"tlx.{ext}").write_text(k.asm[ext])
    print(name, "shared", k.metadata.shared, flush=True)

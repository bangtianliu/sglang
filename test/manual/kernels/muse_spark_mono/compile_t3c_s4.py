"""CPU-only S4 exact-target projection/row/slot compilation before GPU testing."""

import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_projection
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_route as s

root = Path("/home/bangtliu/coord/logs/phaseT-ir/T3c-S4-compile")
root.mkdir(exist_ok=True)
for name, fn, sig, const, w in [
    (
        "projection",
        tlx_projection.project,
        dict(X="*bf16", W="*bf16", P="*i32", Status="*i32", Failure="*i32"),
        dict(
            N=4608,
            K=8192,
            SK=8,
            STAGE=4,
            ROUND_OUTPUT=False,
            PINNED=True,
            POST_BARRIER=True,
            SKIP=-1,
            Seed=None,
            Packed=None,
        ),
        2,
    ),
    (
        "rows",
        s.finish_rows,
        dict(
            P="*i32",
            Bias="*fp32",
            Norm="*fp32",
            Scores="*fp32",
            Ids="*i32",
            Y="*bf16",
            Producer="*i32",
            Rows="*i32",
            Failure="*i32",
        ),
        dict(Seed=None),
        4,
    ),
    (
        "slots",
        s.slots,
        dict(
            Ids="*i32",
            Sorted="*i32",
            Experts="*i32",
            NumPost="*i32",
            Rows="*i32",
            Status="*i32",
            Failure="*i32",
        ),
        dict(LOCAL_START=0, GRID=768, Seed=None),
        4,
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
        options=dict(num_warps=w, enable_fp_fusion=False),
    )
    folder = root / name
    folder.mkdir(exist_ok=True)
    for ext in ("ttgir", "llir", "amdgcn"):
        (folder / f"tlx.{ext}").write_text(k.asm[ext])
    print(json.dumps(dict(phase=name, lds=k.metadata.shared)), flush=True)

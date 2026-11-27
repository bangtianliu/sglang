"""CPU-only compile all TLX S5 phases, including native hardware FP4 upcast."""

import argparse
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective_fast import (
    oneshot_plan,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_experts as s

parser = argparse.ArgumentParser()
parser.add_argument("--scale-async", action="store_true")
parser.add_argument(
    "--out", default="/home/bangtliu/coord/logs/phaseT-ir/T3c-S5-fixed-compile"
)
args = parser.parse_args()
root = Path(args.out)
root.mkdir(exist_ok=True)
p = oneshot_plan(4096, 8, 4)
payload, flags = [
    (r.base, r.rank_stride, r.parity_stride)
    for r in [p.region("payload"), p.region("flags")]
]
cases = []
for first in (True, False):
    cases.append(
        (
            "w13" if first else "w2",
            s.expert_phase,
            dict(
                X="*bf16",
                Sorted="*i32",
                Experts="*i32",
                NumPost="*i32",
                W="*u8",
                Scale="*u8",
                Out="*bf16",
                Status="*i32",
                Failure="*i32",
            ),
            dict(
                GRID=1728,
                FIRST=first,
                SKIP=-1,
                Seed=None,
                WEIGHT_CACHE="",
                SCALE_ASYNC=args.scale_async,
            ),
            9,
        )
    )
cases += [
    (
        "norm",
        s.norm_phase,
        dict(
            Y="*bf16",
            Gamma="*fp32",
            Ids="*i32",
            Scaled="*fp32",
            Inv="*fp32",
            Status="*i32",
            Failure="*i32",
        ),
        dict(RANK=0, Seed=None),
        0,
    ),
    (
        "sum_AR",
        s.sum_reduce,
        dict(
            Ids="*i32",
            Scores="*fp32",
            Scaled="*fp32",
            Inv="*fp32",
            Weighted="*fp32",
            Partial="*bf16",
            Scratch="*i32",
            Peers="*i64",
            Reduced="*fp32",
            Epochs="*i32",
            Status="*i32",
            Failure="*i32",
            Out="*bf16",
        ),
        dict(GRID=8, RANK=0, PAYLOAD=payload, FLAGS=flags, Seed=None),
        0,
    ),
]
for name, fn, sig, const, hint in cases:
    src = ASTSource(
        fn,
        signature=sig,
        constexprs=const,
        attrs={
            (i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
            for i in range(len(sig))
        },
    )
    opts = dict(num_warps=4, enable_fp_fusion=False)
    if hint:
        opts["waves_per_eu"] = hint
    k = triton.compile(src, target=GPUTarget("hip", "gfx1250", 32), options=opts)
    if name in ("norm", "sum_AR"):
        # Regression: FP32 workspace ABI must not numerically convert i32 bits.
        assert "sitofp " not in k.asm["llir"], name
    folder = root / name
    folder.mkdir(exist_ok=True)
    for ext in ("ttir", "ttgir", "llir", "amdgcn"):
        (folder / f"tlx.{ext}").write_text(k.asm[ext])
    print(
        json.dumps(
            dict(
                phase=name,
                shared=k.metadata.shared,
                fp4_convert_sites=k.asm["amdgcn"].count("v_cvt_scale_pk8_bf16_fp4"),
                resource_lines=[
                    l.strip()
                    for l in k.asm["amdgcn"].splitlines()
                    if any(
                        x in l
                        for x in [
                            ".vgpr_count:",
                            ".vgpr_spill_count:",
                            ".private_segment_fixed_size:",
                        ]
                    )
                ],
            )
        ),
        flush=True,
    )

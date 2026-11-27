"""CPU-only explicit-target TLX S2 descriptors, fallback and merge validation."""

import argparse
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_attention as stage,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out",
    type=Path,
    default=Path("/home/bangtliu/coord/logs/phaseT-ir/T3-S2-compile"),
)
root = parser.parse_args().out
root.mkdir(exist_ok=True)
for label, splits, left, has_map, map_page, maxpages in [
    ("full", 128, -1, False, False, 4),
    ("sliding-page", 32, 2047, True, True, 8),
    ("sliding-row", 32, 2047, True, False, 8),
]:
    sig = dict(
        Q="*bf16",
        K="*fp8e4nv",
        V="*fp8e4nv",
        Pages="*i32",
        Map="*i64",
        Length="*i32",
        KS="*fp32",
        VS="*fp32",
        P="*i32",
        LSE="*i32",
        Epochs="*i32",
        Status="*i32",
        Failure="*i32",
    )
    const = dict(
        SPLITS=splits,
        MAX_PAGES=maxpages,
        SCALE=0.125,
        LEFT=left,
        HAS_MAP=has_map,
        MAP_PAGE=map_page,
        POOL_TOKENS=8192,
        SKIP=-1,
        PIPELINED=maxpages > 8,
        DIRECT_KV=False,
        Seed=None,
    )
    for phase, fn, signature, constants in [
        ("partial", stage.partials, sig, const),
        (
            "merge",
            stage.merge,
            dict(
                P="*i32",
                LSE="*i32",
                Status="*i32",
                CtaStatus="*i32",
                Out="*bf16",
                Failure="*i32",
            ),
            dict(SPLITS=splits, Seed=None, CACHE=".cg"),
        ),
    ]:
        src = ASTSource(
            fn,
            signature=signature,
            constexprs=constants,
            attrs={
                (i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
                for i in range(len(signature))
            },
        )
        k = triton.compile(
            src,
            target=GPUTarget("hip", "gfx1250", 32),
            options=dict(num_warps=4, enable_fp_fusion=False),
        )
        folder = root / label / phase
        folder.mkdir(parents=True, exist_ok=True)
        for ext in ("ttgir", "llir", "amdgcn"):
            (folder / f"tlx.{ext}").write_text(k.asm[ext])
        print(
            json.dumps(dict(label=label, phase=phase, shared=k.metadata.shared)),
            flush=True,
        )

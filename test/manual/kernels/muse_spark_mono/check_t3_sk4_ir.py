"""CPU-only check that adding SK8 retains the accepted SK4 S6 instructions."""

import argparse
import collections
import json
import re
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_projection import (
    project,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out",
    type=Path,
    default=Path("/home/bangtliu/coord/logs/phaseT-ir/T3-SK4-compat"),
)
root = parser.parse_args().out
root.mkdir(exist_ok=True)
src = ASTSource(
    project,
    signature=dict(X="*bf16", W="*bf16", P="*i32", Status="*i32", Failure="*i32"),
    constexprs=dict(
        N=8192,
        K=4096,
        STAGE=6,
        SK=4,
        PINNED=True,
        POST_BARRIER=True,
        ROUND_OUTPUT=True,
        SKIP=-1,
        Seed=None,
        Packed=None,
    ),
    attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(5)},
)
k = triton.compile(
    src,
    target=GPUTarget("hip", "gfx1250", 32),
    options=dict(num_warps=2, enable_fp_fusion=False),
)
for ext in ("ttgir", "llir", "amdgcn"):
    (root / f"tlx.{ext}").write_text(k.asm[ext])
old = Path(
    "/home/bangtliu/coord/logs/phaseT-ir/T2-chain-S6-pinned/projection/tlx.amdgcn"
).read_text()


def ops(s):
    return collections.Counter(re.findall(r"^\s*([a-z][a-z0-9_]+)(?:\s|$)", s, re.M))


a = ops(old)
b = ops(k.asm["amdgcn"])
diff = {key: [a[key], b[key]] for key in a.keys() | b.keys() if a[key] != b[key]}
result = dict(
    same_opcode_inventory=not diff,
    differences=diff,
    wmma=b["v_wmma_f32_16x16x32_bf16"],
    tdm=b["tensor_load_to_lds"],
    shared=k.metadata.shared,
)
assert result["wmma"] == 128 and result["tdm"] == 16 and result["shared"] == 133088
(root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))

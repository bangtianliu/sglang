"""CPU-only isolate require_layout pin round trips from release/JIT boundaries."""

import json
import re
from pathlib import Path

import triton
import triton.language as tl
import utlx_plugin as tlx
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_attention import (
    MERGE,
)


@triton.jit
def fixed(value, layout: tl.constexpr):
    return tlx.release_layout(tlx.require_layout(value, layout, pin=True))


@triton.jit
def staged(value):
    value = tlx.require_layout(value, MERGE)
    return tlx.require_layout(value, MERGE, pin=True)


@triton.jit
def carrier(value):
    return tlx.require_layout(value, MERGE, pin=True)


@triton.jit
def probe(X, Y, MODE: tl.constexpr):
    r = tl.arange(0, 128)
    c = tl.arange(0, 64)
    raw = tl.load(X + r[:, None] * 64 + c[None, :])
    if MODE == 7:
        raw = tlx.require_layout(raw, MERGE)
    if MODE == 8:
        x = staged(raw)
    else:
        x = tlx.require_layout(raw, MERGE, pin=True)
    if MODE == 4 or MODE == 6:
        x = tlx.release_layout(x)
    y = x * x
    if MODE == 0:
        y = tlx.require_layout(y, MERGE, pin=True)
    elif MODE == 1:
        y = fixed(y, MERGE)
    elif MODE == 2:
        y = tlx.release_layout(tlx.require_layout(y, MERGE, pin=True))
    elif MODE == 3 or MODE == 5:
        y = carrier(y)
    elif MODE == 4:
        y = fixed(y, MERGE)
    elif MODE == 6 or MODE == 7:
        y = tlx.require_layout(y, MERGE, pin=True)
    elif MODE == 8:
        y = staged(y)
    result = tl.sum(y, 0)
    tl.store(Y + c, result)


root = Path("/home/bangtliu/coord/logs/phaseT-ir/T3b-roundtrip")
root.mkdir(exist_ok=True)
rows = []
for mode in range(9):
    src = ASTSource(
        probe,
        signature=dict(X="*fp32", Y="*fp32"),
        constexprs=dict(MODE=mode),
        attrs={
            (i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(2)
        },
    )
    k = triton.compile(
        src,
        target=GPUTarget("hip", "gfx1250", 32),
        options=dict(num_warps=4, enable_fp_fusion=False),
    )
    folder = root / str(mode)
    folder.mkdir(exist_ok=True)
    for ext in ("ttir", "ttgir", "llir", "amdgcn"):
        (folder / f"probe.{ext}").write_text(k.asm[ext])
    ir = k.asm["ttgir"]
    asm = k.asm["amdgcn"]
    rows.append(
        dict(
            mode=mode,
            conversions=[
                l.strip()
                for l in ir.splitlines()
                if "convert_layout" in l or "require_layout" in l
            ],
            shared=k.metadata.shared,
            barriers=len(re.findall(r"^\s*s_barrier_signal\s", asm, re.M)),
        )
    )
(root / "result.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))

assert rows[4]["barriers"] > rows[1]["barriers"]
assert rows[7]["barriers"] < rows[0]["barriers"]
assert rows[7]["shared"] == 1024

"""Compile the debug guard LLVM with dummy addresses, without initializing a GPU."""

import json
from pathlib import Path

import triton
from prepare_t2_isolation import fault_module
from range_guard_t2 import guard_ir
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.cache import get_override_manager

root = Path("/home/bangtliu/coord/logs/phaseT2-isolation-cpu")
root.mkdir(exist_ok=True)
knobs.cache.dir = str(root / "cache")
knobs.cache.override_dir = str(root / "override")
knobs.compilation.override = True
knobs.compilation.always_compile = True
fn = fault_module().finish
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
)
src = ASTSource(
    fn,
    signature=signature,
    constexprs=dict(Failure=None, PRE_BF16=False, PIN_LAYOUT=True),
    attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(9)},
)
opts = dict(num_warps=4, enable_fp_fusion=False)
target = GPUTarget("hip", "gfx1250", 32)
k = triton.compile(src, target=target, options=opts)
bounds = {
    i: (0x1000000000 + i * 0x100000, b)
    for i, b in enumerate(
        (262144, 262144, 16384, 16384, 32768, 32768, 131072, 262144, 2048)
    )
}
text, sites = guard_ir(
    k.asm["llir"], bounds, k.metadata.shared, 0x2000000000, 0x3000000000
)
(root / "original.llir").write_text(k.asm["llir"])
(root / "guarded.llir").write_text(text)
(root / "sites.json").write_text(json.dumps(sites, indent=2) + "\n")
override = get_override_manager(src.hash())
Path(override.cache_dir).mkdir(parents=True, exist_ok=True)
Path(override.cache_dir, fn.__name__ + ".llir").write_text(text)
guarded = triton.compile(src, target=target, options=opts)
(root / "guarded.amdgcn").write_text(guarded.asm["amdgcn"])
print(
    json.dumps(
        {
            "sites": len(sites),
            "global": sum(s["space"] == 1 for s in sites),
            "lds": sum(s["space"] == 3 for s in sites),
            "guarded_compiled": True,
        }
    ),
    flush=True,
)

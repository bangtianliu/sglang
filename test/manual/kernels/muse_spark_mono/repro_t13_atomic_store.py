"""CPU-only reproducer: uTLX lacks the stock tt.atomic_store conversion.

Run with HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=. No allocation or launch.
The compute carrier is released before the store; no store/pointer pin is used.
"""

import argparse
from pathlib import Path

import triton
import triton.language as tl
import utlx_plugin as tlx
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
)


@triton.jit
def tagged_store(Src, Dst, Tag):
    i = tl.arange(0, 128)
    payload = tlx.require_layout(tl.load(Src + i), ROW1)
    pair = (Tag.to(tl.uint32).to(tl.uint64) << 32) | payload.to(tl.uint64)
    tl.atomic_store(Dst + i, tlx.release_layout(pair), sem="relaxed", scope="gpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ASTSource(
        tagged_store, signature={"Src": "*u32", "Dst": "*u64", "Tag": "u32"}
    )
    kernel = triton.compile(
        source, target=GPUTarget("hip", "gfx1250", 32), options={"num_warps": 4}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("ttgir", "llir", "amdgcn"):
        (args.output / ("tagged_store." + ext)).write_text(kernel.asm[ext])
    asm = kernel.asm["amdgcn"]
    assert "global_wb" not in asm and "global_inv" not in asm
    assert "global_store_b64" in asm
    print("PASS: compiler-tracked relaxed paired64 store; no GPU execution")


if __name__ == "__main__":
    main()

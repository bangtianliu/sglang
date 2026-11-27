"""Compile-first role specialization feasibility; never launch unverified barriers."""

import json
from pathlib import Path

import triton
import triton.language as tl
import utlx_plugin as tlx
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_core import (
    expert_tile,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_overlap import (
    warm_weights,
)


@triton.jit
def role_pair(X, Sorted, Experts, W13, S13, W2, S2, Out, Checks):
    task = tl.program_id(0)
    block = task // 64
    expert = tl.load(Experts + block)
    with tlx.async_tasks():
        with tlx.async_task("default"):
            expert_tile(X, Sorted, W13, S13, Out, block, expert, task % 64, True)
        with tlx.async_task(num_warps=4):
            checksum = warm_weights(W2, S2, expert, task % 64)
            tl.store(Checks + task, checksum)


def main():
    signature = {
        k: (
            "*bf16"
            if k in ("X", "Out")
            else "*u8"
            if k in ("W13", "S13", "W2", "S2")
            else "*i32"
        )
        for k in role_pair.arg_names
    }
    path = Path("/home/bangtliu/coord/logs/phaseT-ir/T9-roles")
    path.mkdir(exist_ok=True)
    result = dict(launched=False)
    try:
        kernel = triton.compile(
            ASTSource(
                role_pair,
                signature=signature,
                attrs={
                    (i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
                    for i in range(len(signature))
                },
            ),
            target=GPUTarget("hip", "gfx1250", 32),
            options=dict(num_warps=4, enable_fp_fusion=False),
        )
        for ext in ("ttir", "ttgir", "llir", "amdgcn"):
            (path / ("tlx." + ext)).write_text(kernel.asm[ext])
        result.update(
            compiled=True,
            metadata=kernel.metadata._asdict(),
            note="Do not launch until default-region workgroup barriers are proved partition-scoped.",
        )
    except Exception as e:
        result.update(
            compiled=False,
            error_type=type(e).__name__,
            error=str(e),
            note="Bounded backend capability probe; no GPU kernel launched. No compiler changes in this phase.",
        )
    Path("/home/bangtliu/coord/logs/phaseT9-roles-feasibility.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()

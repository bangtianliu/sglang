"""Readonly GPU0 weight warming; ordered handoffs and cache policies stay intact."""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW4,
)


@triton.jit
def readonly(
    X, Checks, N: tl.constexpr, GRID: tl.constexpr, HEAD_SLEEP: tl.constexpr = 64
):
    pid = tl.program_id(0)
    # Bounded delay keeps warm traffic off the preceding kernel's initial loads.
    for _ in range(HEAD_SLEEP):
        tl.inline_asm_elementwise(
            "s_sleep 1\nv_mov_b32 $0,0", "=v", [], dtype=tl.int32, is_pure=False, pack=1
        )
    i = tlx.require_layout(tl.arange(0, 4096), ROW4)
    checksum = tl.full((4096,), 0, tl.uint32)
    for block in range(pid, tl.cdiv(N, 4096), GRID):
        at = block * 4096 + i
        value = tl.load(X + at, at < N, 0)
        checksum += value.to(tl.uint16, bitcast=True).to(tl.uint32)
    tl.store(Checks + pid, tl.sum(checksum, 0))

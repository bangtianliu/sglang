"""Manual-only eight-wave expert/warm role prototype with local compiler bridge.

The bridge is scoped to this kernel's compile/launch. Normal chain kernels keep
buffer ops and their existing compiler pipeline. No installed files are changed.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx
from tlx_ir import dump
from triton import knobs
from triton._C.libtriton import passes

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard as guard,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_core import (
    expert_tile,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_overlap import (
    warm_weights,
)


@contextmanager
def role_compile():
    saved_env = os.environ.get("AMDGCN_USE_BUFFER_OPS")
    saved_knob = knobs.amd.use_buffer_ops
    original = passes.plugin.utlx_lower_pinned_layout

    def allocate_then_pin(pm, *args):
        passes.ttgpuir.add_allocate_warp_groups(pm)
        return original(pm, *args)

    os.environ["AMDGCN_USE_BUFFER_OPS"] = "0"
    knobs.amd.use_buffer_ops = False
    passes.plugin.utlx_lower_pinned_layout = allocate_then_pin
    try:
        yield
    finally:
        passes.plugin.utlx_lower_pinned_layout = original
        knobs.amd.use_buffer_ops = saved_knob
        if saved_env is None:
            os.environ.pop("AMDGCN_USE_BUFFER_OPS", None)
        else:
            os.environ["AMDGCN_USE_BUFFER_OPS"] = saved_env


@triton.jit
def role_expert(
    X,
    Sorted,
    Experts,
    NumPost,
    W,
    Scale,
    Out,
    Status,
    Failure,
    W2,
    S2,
    Checks,
    SKIP: tl.constexpr = -1,
    Seed=None,
):
    pid = tl.program_id(0)
    allowed = guard.enabled(Failure, 5)
    ok = allowed & (pid != SKIP)
    tasks = tl.load(NumPost) // 16 * 64
    guard.delay(Seed, pid, 0)
    if ok:
        for task in range(pid, tasks, 1728):
            block = task // 64
            expert = tl.load(Experts + block)
            with tlx.async_tasks():
                with tlx.async_task("default"):
                    expert_tile(
                        X, Sorted, W, Scale, Out, block, expert, task % 64, True
                    )
                with tlx.async_task(num_warps=4):
                    for _ in range(32):
                        tl.inline_asm_elementwise(
                            "s_sleep 1\nv_mov_b32 $0,0",
                            "=v",
                            [],
                            dtype=tl.int32,
                            is_pure=False,
                            pack=1,
                        )
                    checksum = warm_weights(W2, S2, expert, task % 64)
                    tl.store(Checks + pid, checksum)
    guard.delay(Seed, pid, 1)
    tl.store(Status + pid, ok)
    if allowed:
        guard.finish(Failure, ok, pid, 5)


class RoleAdapter:
    def __init__(self, raw, label):
        self.raw = raw
        self.label = label
        self.saved = None
        self.spaces = {}
        self.admissions = []

    def __getattr__(self, k):
        return getattr(self.raw, k)

    def __getitem__(self, grid):
        def call(*aa, **kw):
            if kw["FIRST"]:
                self.saved = (aa, kw)
                return None
            first, k1 = self.saved
            self.saved = None
            key = (first[7].data_ptr(), k1.get("SKIP", -1), k1.get("Seed") is None)
            if key not in self.spaces:
                checks = torch.empty((1728,), device=first[0].device, dtype=torch.int32)
                args = (*first, aa[4], aa[5], checks)
                options = dict(
                    SKIP=k1.get("SKIP", -1),
                    Seed=k1.get("Seed"),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                with role_compile():
                    kernel = role_expert.warmup(*args, **options, grid=(1728,))
                r = kernel_occupancy("T9-role-expert", kernel)
                assert kernel.metadata.num_warps == 8 and kernel.n_spills == 0
                self.admissions.append(
                    dict(
                        resources=dump(kernel, self.label + "/role", "tlx"),
                        grid=1728,
                        scope_note="W13 still completes before default-cache W2; readonly warming has no flag handoff. Process-local compiler bridge only.",
                    )
                )
                Path(
                    "/home/bangtliu/coord/logs", self.label + "-admission.json"
                ).write_text(json.dumps(self.admissions, indent=2) + "\n")
                self.spaces[key] = (args, options)
            args, options = self.spaces[key]
            with role_compile():
                role_expert[(1728,)](*args, **options)
            return self.raw[grid](*aa, **kw)

        return call

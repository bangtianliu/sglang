"""GPU0 tiny async_tasks proof with CPU watchdog and copied progress records.

Compiler workaround is process-local. Production images and plugin files stay
unchanged. No trap, peer, mapped-host kernel pointer or unbounded GPU poll.
"""

import json
import os
import time
from pathlib import Path

# Keep this probe's altered pipeline/cache key separate from production kernels.
os.environ["AMDGCN_USE_BUFFER_OPS"] = "0"
import torch
import triton
import triton.language as tl
import utlx_plugin as tlx
from layer_fixture import require_gpu0
from tlx_ir import dump
from triton._C.libtriton import passes

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW4,
)

original = passes.plugin.utlx_lower_pinned_layout


def allocate_then_pin(pm, *args):
    passes.ttgpuir.add_allocate_warp_groups(pm)
    return original(pm, *args)


passes.plugin.utlx_lower_pinned_layout = allocate_then_pin


@triton.jit
def roles(X, Y, Out, Progress):
    with tlx.async_tasks():
        with tlx.async_task("default"):
            tl.store(Progress, 1, cache_modifier=".cg")
            i = tlx.require_layout(tl.arange(0, 4096), ROW4)
            value = tl.sum(tl.load(X + i).to(tl.float32), 0)
            tl.store(Out, value)
            tl.store(Progress + 1, 1, cache_modifier=".cg")
        with tlx.async_task(num_warps=4):
            tl.store(Progress + 2, 1, cache_modifier=".cg")
            j = tlx.require_layout(tl.arange(0, 4096), ROW4)
            value2 = tl.sum(tl.load(Y + j).to(tl.float32), 0)
            tl.store(Out + 1, value2)
            tl.store(Progress + 3, 1, cache_modifier=".cg")


def main():
    require_gpu0()
    path = Path("/home/bangtliu/coord/logs/phaseT9-roles-runtime.json")
    records = []
    x = torch.ones((4096,), device="cuda", dtype=torch.bfloat16)
    y = torch.full_like(x, 2)
    out = torch.empty((2,), device="cuda")
    progress = torch.zeros((4,), device="cuda", dtype=torch.int32)
    kernel = roles.warmup(
        x, y, out, progress, num_warps=4, enable_fp_fusion=False, grid=(1,)
    )
    assert kernel.metadata.num_warps == 8
    resource = dump(kernel, "T9-roles-runtime", "tlx")
    assert kernel.n_spills == 0
    print(
        "COMPILED",
        resource["vgprs"],
        resource["waves"],
        "progress_address",
        progress.data_ptr(),
        flush=True,
    )
    transfer = torch.cuda.Stream()
    host = torch.empty((4,), dtype=torch.int32, pin_memory=True)
    try:
        for step in range(4):
            progress.zero_()
            out.fill_(float("nan"))
            end = torch.cuda.Event()
            roles[(1,)](x, y, out, progress, num_warps=4, enable_fp_fusion=False)
            end.record()
            deadline = time.monotonic() + 10
            snapshots = []
            while not end.query() and time.monotonic() < deadline:
                with torch.cuda.stream(transfer):
                    host.copy_(progress, non_blocking=True)
                    copied = torch.cuda.Event()
                    copied.record()
                until = time.monotonic() + 0.05
                while not copied.query() and time.monotonic() < until:
                    time.sleep(0.001)
                if copied.query():
                    snapshots.append(host.tolist())
                time.sleep(0.005)
            if not end.query():
                path.write_text(
                    json.dumps(
                        dict(
                            pass_=False,
                            reason="10s CPU deadline",
                            step=step,
                            snapshots=snapshots,
                            progress_address=progress.data_ptr(),
                            resources=resource,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                print("TIMEOUT: no retry", flush=True)
                os._exit(42)
            torch.cuda.synchronize()
            assert progress.eq(1).all() and torch.equal(
                out, torch.tensor([4096.0, 8192.0], device="cuda")
            ), (progress.tolist(), out.tolist())
            records.append(
                dict(
                    step=step,
                    progress=progress.tolist(),
                    output=out.tolist(),
                    snapshots=snapshots,
                )
            )
            print("PASS roles", step, flush=True)
    except Exception as e:
        path.write_text(
            json.dumps(
                dict(
                    pass_=False,
                    error=repr(e),
                    records=records,
                    progress_address=progress.data_ptr(),
                    resources=resource,
                ),
                indent=2,
            )
            + "\n"
        )
        print(repr(e), flush=True)
        os._exit(43)
    path.write_text(
        json.dumps(
            dict(
                pass_=True,
                records=records,
                resources=resource,
                note="Only this process disables buffer ops and inserts allocation. No production compiler/image edits.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

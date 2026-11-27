"""Isolate gfx1250 device cache maintenance; GPU0, timeout -s KILL 300."""

import json
from pathlib import Path

import numpy as np
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync

LOG = Path("/home/bangtliu/coord/logs")


@gluon.jit
def dirty(Data, Value, SIZE: gl.constexpr):
    i = gl.program_id(0) * 1024 + gl.arange(
        0, 1024, layout=gl.BlockedLayout([1], [32], [1], [0])
    )
    gl.store(Data + i, Value, i < SIZE)


@gluon.jit
def fences(Out, REPS: gl.constexpr, KIND: gl.constexpr):
    start = sync.clock()
    for _ in range(REPS):
        if KIND == "wb":
            gl.inline_asm_elementwise(
                "global_wb scope:SCOPE_DEV\ns_wait_storecnt 0\nv_mov_b32 $0, 0",
                "=v",
                [],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
        elif KIND == "inv":
            gl.inline_asm_elementwise(
                "global_inv scope:SCOPE_DEV\ns_wait_loadcnt 0\nv_mov_b32 $0, 0",
                "=v",
                [],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
        else:
            gl.inline_asm_elementwise(
                "v_mov_b32 $0, 0", "=v", [], dtype=gl.int32, is_pure=False, pack=1
            )
    end = sync.clock()
    gl.store(Out + gl.program_id(0), end - start)


def main():
    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.startswith("gfx1250")
    size = 64 * 1024 * 1024 // 4
    data = torch.empty(size, device="cuda", dtype=torch.int32)
    out = torch.zeros(props.multi_processor_count, device="cuda", dtype=torch.int64)
    rows = []
    for grid in (1, props.multi_processor_count):
        for state in ("clean", "dirty64MiB"):
            for kind in ("none", "wb", "inv"):
                for reps in (1, 8, 32):
                    values = []
                    for sample in range(20):
                        dirty[(size // 1024,)](data, sample, SIZE=size, num_warps=1)
                        if state == "clean":
                            fences[(1,)](out, REPS=1, KIND="wb", num_warps=1)
                        kernel = fences[(grid,)](out, REPS=reps, KIND=kind, num_warps=1)
                        values.append(out[:grid].cpu().numpy().copy() / 100)
                    asm = kernel.asm["amdgcn"]
                    (LOG / f"phase2.2c-{kind}-N{reps}.amdgcn").write_text(asm)
                    values = np.asarray(values)
                    row = dict(
                        grid=grid,
                        state=state,
                        kind=kind,
                        repetitions=reps,
                        median_us=float(np.median(values)),
                        max_us=float(values.max()),
                        median_per_instruction_us=float(np.median(values)) / reps,
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    (LOG / "phase2.2c-fences.json").write_text(
                        json.dumps(rows, indent=2)
                    )


if __name__ == "__main__":
    main()

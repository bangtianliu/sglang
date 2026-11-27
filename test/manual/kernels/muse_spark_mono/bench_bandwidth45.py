"""Read bandwidth ramp: conventional grid and resident 768-CTA grid/barrier."""

import argparse
import functools
import json

import torch
import triton
from layer_fixture import require_gpu0
from mono_case45 import LOG
from paired45 import measure
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
    CommonShape,
)


@g.jit
def read(
    X,
    Y,
    State,
    Status,
    N: gl.constexpr,
    GRID: gl.constexpr,
    PERSIST: gl.constexpr,
    BARRIER: gl.constexpr,
):
    pid = gl.program_id(0)
    ok = gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    if BARRIER:
        ok, polls, _ = scoped.counter_barrier(
            State, 1, 1000000, GRID, 32, MAX_SPINS=65536
        )
        gl.store(Status + pid * 2 + 1, polls)
    gl.store(Status + pid * 2, ok)
    i = gl.arange(0, 4096, layout=gl.BlockedLayout([4], [32], [4], [0]))
    total = gl.full((4096,), 0, gl.float32, i.type.layout)
    if ok:
        if PERSIST:
            for block in range(pid, gl.cdiv(N, 4096), GRID):
                at = block * 4096 + i
                total += gl.load(X + at, at < N, 0)
        else:
            at = pid * 4096 + i
            total = gl.load(X + at, at < N, 0)
    gl.store(Y + pid, gl.sum(total, 0))


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--bytes", type=int, required=True)
    a = p.parse_args()
    n = a.bytes // 4
    assert n * 4 == a.bytes
    cases = {}
    meta = {}
    for label in ("wide", "persistent", "barrier"):
        grid = triton.cdiv(n, 4096) if label == "wide" else 768
        x = torch.ones((n,), device="cuda")
        y = torch.empty((grid,), device="cuda")
        state = torch.zeros(((24 + 1 + 768) * 32,), device="cuda", dtype=torch.int32)
        status = torch.empty((grid, 2), device="cuda", dtype=torch.int32)
        pid = torch.arange(grid, device="cuda", dtype=torch.int64)
        blocks = triton.cdiv(n, 4096)
        count = (
            torch.clamp((blocks - 1 - pid) // grid + 1, min=0)
            if label != "wide"
            else torch.ones_like(pid)
        )
        expected = (
            count * 4096 - torch.where(pid == (blocks - 1) % grid, blocks * 4096 - n, 0)
        ).float()
        call = functools.partial(
            read[(grid,)],
            x,
            y,
            state,
            status,
            n,
            grid,
            label != "wide",
            label == "barrier",
            **CommonShape().options(),
        )
        compiled = read.warmup(
            x,
            y,
            state,
            status,
            n,
            grid,
            label != "wide",
            label == "barrier",
            grid=(grid,),
            **CommonShape().options(),
        )
        compiled._init_handles()
        if label != "wide":
            CommonShape().admit(label, compiled, grid=grid)

        def check(y=y, expected=expected, status=status, label=label):
            assert torch.equal(y, expected), (label, "read checksum")
            assert bool(status[:, 0].eq(1).all()), (label, "barrier failure")

        cases[label] = (state.zero_, call, check)
        meta[label] = dict(
            grid=grid,
            vgpr=compiled.n_regs,
            spills=compiled.n_spills,
            lds=compiled.metadata.shared,
        )
    records = []
    for other in ("wide", "persistent"):
        result, _ = measure({other: cases[other], "barrier": cases["barrier"]})
        for name, v in result.items():
            v["read_TB_s"] = a.bytes / v["median_us"] / 1e6
        records.append(dict(pair=other + "-barrier", results=result))
        print(
            json.dumps(
                dict(
                    bytes=a.bytes,
                    pair=other,
                    medians={
                        k: dict(us=v["median_us"], TB_s=v["read_TB_s"])
                        for k, v in result.items()
                    },
                )
            ),
            flush=True,
        )
    (LOG / f"phase45-bandwidth-{a.bytes}.json").write_text(
        json.dumps(dict(bytes=a.bytes, resources=meta, records=records), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()

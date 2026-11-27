"""GPU0 row epilogues alone: ready inputs, no waits, exact pre4.4d reference."""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from layer_fixture import require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
    CommonShape,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_tile_words,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    boundary,
    producer_rows,
)

ROOT = Path("/home/bangtliu/coord")
LOG = ROOT / "logs"
spec = importlib.util.spec_from_file_location(
    "boundary_before44d", ROOT / "phase44d-boundary-before.py"
)
before = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = before
spec.loader.exec_module(before)


@g.jit
def run(
    Branch,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    State,
    Out,
    Updated,
    Times,
    STAGE: gl.constexpr,
    WHICH: gl.constexpr,
    ROWS: gl.constexpr,
    DETAIL: gl.constexpr = False,
    ORIGINAL: gl.constexpr = False,
):
    for rr in range(ROWS):
        row = gl.program_id(0) * ROWS + rr
        T = Times + row * 32 if DETAIL else None
        if DETAIL:
            gl.store(T + 16, sync.clock())
        if STAGE == 3 and WHICH == 0:
            i = gl.arange(0, 8192, layout=gl.BlockedLayout([1], [32], [4], [0]))
            value = load_tile_words(Branch + row * 8192 + i, i >= 0, 8192).to(
                gl.float32, bitcast=True
            )
        else:
            value = None
        if DETAIL:
            gl.store(T + 17, sync.clock())
        if WHICH == 0:
            if ORIGINAL:
                before.norm_residual(
                    row,
                    Branch,
                    Stream,
                    Post if STAGE == 6 else None,
                    Pre,
                    Alpha,
                    Beta,
                    Out,
                    Updated,
                    STAGE == 6,
                    False,
                    True,
                    Value=value,
                    SCOPED_IO=True,
                )
            else:
                boundary.norm_residual(
                    row,
                    Branch,
                    Stream,
                    Post if STAGE == 6 else None,
                    Pre,
                    Alpha,
                    Beta,
                    Out,
                    Updated,
                    STAGE == 6,
                    False,
                    True,
                    Trace=T,
                    Value=value,
                    SCOPED_IO=True,
                    PACKED_ROW=True,
                )
        else:
            producer_rows.finish_row(
                State,
                row,
                1,
                Stream,
                Post if STAGE == 6 else None,
                Pre,
                Alpha,
                Beta,
                Out,
                Updated,
                8192,
                False,
                Trace=T,
            )
        if DETAIL:
            gl.store(T + 18, sync.clock())


def stats(k):
    ins = []
    for l in k.asm["amdgcn"].splitlines():
        m = re.match(
            r"\s*((?:global|flat|buffer|scratch)_(?:load|store)_\w+|s_wait_\w+)\b", l
        )
        if m:
            ins.append(m[1])
    return dict(
        vgpr=k.n_regs,
        spills=k.n_spills,
        lds=k.metadata.shared,
        static_instructions=dict(Counter(ins)),
    )


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=(3, 6), required=True)
    p.add_argument("--label", default="before")
    a = p.parse_args()
    torch.manual_seed(44)
    branch = torch.randn((8, 8192), device="cuda")
    stream = torch.randn_like(branch)
    post = (1 + 0.05 * torch.randn((8192,), device="cuda")).bfloat16()
    pre = (1 + 0.05 * torch.randn_like(post.float())).bfloat16()
    alpha = torch.rand_like(post.float())
    beta = torch.rand_like(alpha)
    state = producer_rows.make_state("cuda", 8192)
    v = branch.bfloat16().float()
    if a.stage == 6:
        v = (v * post.float()).bfloat16().float()
    state[: 8 * 8192].copy_(v.reshape(-1).view(torch.int32))
    at = producer_rows.plan(8192)
    state[at[1] : at[1] + 8 * 8192].copy_((v * v).reshape(-1).view(torch.int32))
    state[at[9] : at[9] + 8 * 32 : 32] = 1
    out = torch.empty_like(branch, dtype=torch.bfloat16)
    updated = torch.empty_like(branch)
    times = torch.zeros((8, 32), device="cuda", dtype=torch.uint64)
    args = (
        branch.view(torch.int32),
        stream,
        post,
        pre,
        alpha,
        beta,
        state,
        out,
        updated,
        times,
    )
    shape = CommonShape()
    opts = shape.options()
    records = []
    for grid, nrows in ((1, 1), (1, 8), (8, 1)):
        names = dict(STAGE=a.stage, ROWS=nrows)
        refk = run.warmup(*args, **names, WHICH=0, ORIGINAL=True, grid=(grid,), **opts)
        shape.admit("row44d-reference", refk, grid=grid)
        run[(grid,)](*args, **names, WHICH=0, ORIGINAL=True, **opts)
        torch.cuda.synchronize()
        ref = (out.clone(), updated.clone())
        for which in (0, 1):
            prefix = f"phase44d-{a.label}-s{a.stage}-w{which}-g{grid}-r{nrows}"
            call = lambda: run[(grid,)](*args, **names, WHICH=which, **opts)
            k = run.warmup(*args, **names, WHICH=which, grid=(grid,), **opts)
            shape.admit(prefix, k, grid=grid)
            for fmt in ("amdgcn", "llir"):
                (LOG / f"{prefix}.{fmt}").write_text(k.asm[fmt])
            call()
            torch.cuda.synchronize()
            for name, actual, expect in zip(("output", "stream"), (out, updated), ref):
                typ = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
                assert torch.equal(
                    actual[: grid * nrows].view(typ), expect[: grid * nrows].view(typ)
                ), (prefix, name)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(32):
                    call()
            us = []
            for rep in range(35):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                if rep >= 5:
                    us.append(start.elapsed_time(end) * 1000 / 32)
            us.sort()
            for name, actual, expect in zip(("output", "stream"), (out, updated), ref):
                typ = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
                assert torch.equal(
                    actual[: grid * nrows].view(typ), expect[: grid * nrows].view(typ)
                ), (prefix, name, "after timing")
            kd = run.warmup(
                *args, **names, WHICH=which, DETAIL=True, grid=(grid,), **opts
            )
            shape.admit(prefix + "-detail", kd, grid=grid)
            times.zero_()
            run[(grid,)](*args, **names, WHICH=which, DETAIL=True, **opts)
            torch.cuda.synchronize()
            for name, actual, expect in zip(("output", "stream"), (out, updated), ref):
                typ = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
                assert torch.equal(
                    actual[: grid * nrows].view(typ), expect[: grid * nrows].view(typ)
                ), (prefix, name, "detail")
            (LOG / f"{prefix}-detail.amdgcn").write_text(kd.asm["amdgcn"])
            record = dict(
                stage=a.stage,
                which=which,
                grid=grid,
                rows_per_cta=nrows,
                median_us=(us[14] + us[15]) / 2,
                p99_us=us[-1],
                bitwise=True,
                resource=stats(k),
                detail_resource=stats(kd),
                clocks=times[: grid * nrows].cpu().tolist(),
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    (LOG / f"phase44d-{a.label}-s{a.stage}.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

"""GPU0 cooperative tiled boundary vs the accepted whole-row device function."""

import json
from pathlib import Path

import torch
from layer_fixture import require_gpu0
from layer_reference import exact_check
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
    CommonShape,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    parallel_rows as rows,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    norm_residual,
)


@g.jit
def legacy(Branch, Stream, Post, Pre, Alpha, Beta, Out, Updated):
    norm_residual(
        gl.program_id(0),
        Branch,
        Stream,
        Post,
        Pre,
        Alpha,
        Beta,
        Out,
        Updated,
        Post is not None,
        False,
        True,
        SCOPED_IO=True,
    )


@g.jit
def tiled(
    Branch,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    State,
    Status,
    Diag,
    Metrics,
    GRID: gl.constexpr,
):
    pid = gl.program_id(0)
    ok = rows.boundary(
        pid,
        1,
        State,
        Branch,
        Stream,
        Post,
        Pre,
        Alpha,
        Beta,
        Out,
        Updated,
        None,
        None,
        Diag,
        Metrics,
        GRID,
        6,
    )
    gl.store(Status + pid, ok)


def main():
    require_gpu0()
    shape = CommonShape()
    grid = shape.grid("cuda")
    records = []
    for post_scale in (False, True):
        gen = torch.Generator(device="cuda").manual_seed(44)
        branch = torch.randn((8, 8192), device="cuda", generator=gen).bfloat16().float()
        stream = torch.randn((8, 8192), device="cuda", generator=gen)
        pre = (1 + torch.randn((8192,), device="cuda", generator=gen) * 0.02).bfloat16()
        post = pre.clone() if post_scale else None
        alpha = torch.full((8192,), 0.87, device="cuda")
        beta = torch.full_like(alpha, 0.42)
        old = torch.empty_like(branch, dtype=torch.bfloat16)
        old_stream = torch.empty_like(branch)
        new = torch.full_like(old, float("nan"))
        new_stream = torch.full_like(branch, float("nan"))
        state = rows.make_state("cuda", 8192)
        p = rows.row_plan(8192)
        state[p[7] : p[7] + 1024 * 32 : 32].fill_(1)
        status = torch.zeros((grid,), device="cuda", dtype=torch.int32)
        diag = torch.zeros((grid, 16), device="cuda", dtype=torch.int64)
        metrics = torch.zeros((grid, 32), device="cuda", dtype=torch.int64)
        a = (
            branch.view(torch.int32),
            stream,
            post,
            pre,
            alpha,
            beta,
            new,
            new_stream,
            state,
            status,
            diag,
            metrics,
        )
        k = tiled.warmup(*a, GRID=grid, **shape.options(), grid=(grid,))
        resource = shape.admit("parallel-boundary-probe", k, grid=grid)
        print(
            json.dumps(
                dict(
                    event="compiled",
                    post=post_scale,
                    vgprs=k.n_regs,
                    spills=k.n_spills,
                    lds=k.metadata.shared,
                    occupancy=resource.active_ctas_per_cu,
                )
            ),
            flush=True,
        )
        legacy[(8,)](
            branch.view(torch.int32),
            stream,
            post,
            pre,
            alpha,
            beta,
            old,
            old_stream,
            **shape.options(),
        )
        tiled[(grid,)](*a, GRID=grid, **shape.options())
        torch.cuda.synchronize()
        exact_check(status, 1, "parallel boundary status", {"_rank": 0})
        if not torch.equal(
            old.view(torch.int16), new.view(torch.int16)
        ) or not torch.equal(
            old_stream.view(torch.int32), new_stream.view(torch.int32)
        ):
            torch.save(
                dict(
                    branch=branch.cpu(),
                    stream=stream.cpu(),
                    post=None if post is None else post.cpu(),
                    pre=pre.cpu(),
                    old=old.cpu(),
                    old_stream=old_stream.cpu(),
                    new=new.cpu(),
                    new_stream=new_stream.cpu(),
                    state=state.cpu(),
                ),
                "/home/bangtliu/coord/logs/phase44-boundary-mismatch.pt",
            )
        exact_check(
            new.view(torch.int16),
            old.view(torch.int16),
            "boundary BF16 bits",
            {"_rank": 0},
        )
        exact_check(
            new_stream.view(torch.int32),
            old_stream.view(torch.int32),
            "boundary FP32 stream bits",
            {"_rank": 0},
        )
        records.append(
            dict(
                post=post_scale,
                bitwise=True,
                active_ctas=int((metrics[:, 21] > 0).sum()),
            )
        )
        print(json.dumps(records[-1]), flush=True)
    Path("/home/bangtliu/coord/logs/phase44-boundary.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

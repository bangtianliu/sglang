"""GPU0 proof that tile statistics keep the accepted full-row FP32 sum tree."""

import json
from pathlib import Path

import torch
from layer_fixture import require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    parallel_rows as rows,
)


@g.jit
def whole(X, Out, N: gl.constexpr):
    r = gl.program_id(0)
    i = gl.arange(0, N, layout=gl.BlockedLayout([1], [32], [4], [0]))
    v = gl.load(X + r * N + i)
    s = gl.sum(v * v, 0)
    gl.store(Out + r * 2, s)
    gl.store(Out + r * 2 + 1, gl.rsqrt(s / N + 1e-5))


@g.jit
def partial(X, Stats, N: gl.constexpr):
    r = gl.program_id(0)
    tile = gl.program_id(1)
    c = rows.columns(tile, N)
    v = gl.load(X + r * N + c)
    s = rows.partial_stat(v)
    lane = gl.arange(0, 8, layout=s.type.layout)
    gl.store(Stats + r * (N // 8) + (tile // 16) * 128 + (tile % 16) * 8 + lane, s)


@g.jit
def finish(Stats, Out, N: gl.constexpr):
    r = gl.program_id(0)
    s, inv = rows.factor(Stats, r, N, 1e-5)
    gl.store(Out + r * 2, s)
    gl.store(Out + r * 2 + 1, inv)


def main():
    require_gpu0()
    records = []
    for n in (4096, 8192):
        gen = torch.Generator(device="cuda").manual_seed(44 + n)
        # Many exponents make an accidental reduction-order change visible.
        x = torch.randn((128, n), device="cuda", generator=gen) * torch.exp2(
            torch.randint(-8, 8, (128, n), device="cuda", generator=gen).float()
        )
        old = torch.empty((128, 2), device="cuda")
        new = torch.empty_like(old)
        stats = torch.empty((128, n // 8), device="cuda")
        whole[(128,)](x, old, n, num_warps=4, enable_fp_fusion=False)
        partial[(128, n // 64)](x, stats, n, num_warps=4, enable_fp_fusion=False)
        finish[(128,)](stats, new, n, num_warps=4, enable_fp_fusion=False)
        torch.cuda.synchronize()
        bad = old.view(torch.int32) != new.view(torch.int32)
        r = dict(
            width=n,
            rows=128,
            bit_mismatches=int(bad.sum()),
            max_abs=float((old - new).abs().max()),
        )
        records.append(r)
        print(json.dumps(r), flush=True)
        if bad.any():
            torch.save(
                dict(x=x.cpu(), old=old.cpu(), new=new.cpu()),
                "/home/bangtliu/coord/logs/phase44-tree-mismatch.pt",
            )
            raise AssertionError(r)
    Path("/home/bangtliu/coord/logs/phase44-tree.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

"""Minimal uTLX row-reduction pin reproducer; explicit-target compile needs no GPU."""

import argparse
import json
from pathlib import Path

import triton
import triton.language as tl
import utlx_plugin as tlx
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


class RowOne(tlx.layout_encoding):
    def __repr__(self):
        return "row_one_4_waves"

    def __eq__(self, other):
        return isinstance(other, RowOne)

    def __hash__(self):
        return hash(RowOne)

    def to_ir(self, builder, *_):
        return builder.get_blocked_layout([1], [32], [4], [0], [])


ROW = tl.constexpr(RowOne())


@triton.jit
def row_norm(X, P, Y, N: tl.constexpr, PIN: tl.constexpr):
    i = tl.arange(0, N)
    x = (
        (tl.load(X + i).to(tl.bfloat16).to(tl.float32) * tl.load(P + i).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    x = tlx.require_layout(x, ROW, pin=PIN)
    squares = tlx.require_layout(x * x, ROW, pin=PIN)
    total = tl.sum(squares, 0)
    y = x * tl.rsqrt(total / N + 1e-5)
    tl.store(Y + i, y.to(tl.bfloat16))


@g.jit
def row_norm_gl(X, P, Y, N: gl.constexpr):
    i = gl.arange(0, N, layout=gl.BlockedLayout([1], [32], [4], [0]))
    x = (
        (gl.load(X + i).to(gl.bfloat16).to(gl.float32) * gl.load(P + i).to(gl.float32))
        .to(gl.bfloat16)
        .to(gl.float32)
    )
    total = gl.sum(x * x, 0)
    gl.store(Y + i, (x * gl.rsqrt(total / N + 1e-5)).to(gl.bfloat16))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    records = {}
    for pin in (False, True):
        src = ASTSource(
            row_norm,
            signature={"X": "*fp32", "P": "*bf16", "Y": "*bf16"},
            constexprs={"N": 8192, "PIN": pin},
            attrs={
                (j,): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
                for j in range(3)
            },
        )
        k = triton.compile(
            src,
            target=GPUTarget("hip", "gfx1250", 32),
            options={"num_warps": 4, "enable_fp_fusion": False},
        )
        name = "pinned" if pin else "unpinned"
        for level in ("ttir", "ttgir", "llir", "amdgcn"):
            (a.out / (name + "." + level)).write_text(k.asm[level])
        records[name] = {
            "metadata": k.metadata._asdict(),
            "layouts": [
                x
                for x in k.asm["ttgir"].splitlines()
                if x.startswith("#") and "blocked<" in x
            ],
        }
    if a.gpu:
        import os

        import torch

        assert (
            os.environ.get("HIP_VISIBLE_DEVICES") == "0"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        )
        assert torch.cuda.device_count() == 1
        generator = torch.Generator(device="cuda").manual_seed(20261118)
        x = torch.randn(8192, device="cuda", dtype=torch.float32, generator=generator)
        scale = torch.randn(
            8192, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        ys = []
        for pin in (False, True):
            y = torch.empty_like(x, dtype=torch.bfloat16)
            row_norm[(1,)](x, scale, y, 8192, pin, num_warps=4, enable_fp_fusion=False)
            ys.append(y)
        torch.cuda.synchronize()
        reference = torch.empty_like(ys[1])
        row_norm_gl[(1,)](
            x, scale, reference, 8192, num_warps=4, enable_fp_fusion=False
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(ys[1], reference, rtol=0, atol=0)
        rounded = (x.bfloat16().float() * scale.float()).bfloat16().float().double()
        expected = rounded * torch.rsqrt(rounded.square().mean() + 1e-5)
        torch.testing.assert_close(ys[1].double(), expected, rtol=0.004, atol=0.004)
        records["gpu"] = {
            "pinned_vs_gluon": "exact",
            "pinned_vs_fp64": "tolerance pass",
            "unpinned_vs_pinned_different_words": int(
                (ys[0].view(torch.int16) != ys[1].view(torch.int16)).sum()
            ),
        }
    (a.out / "summary.json").write_text(
        json.dumps(records, indent=2, default=str) + "\n"
    )
    print(json.dumps(records, default=str), flush=True)


if __name__ == "__main__":
    main()

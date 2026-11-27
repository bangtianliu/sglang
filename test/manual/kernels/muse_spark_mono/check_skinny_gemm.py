"""Verify fixed-order skinny core independently of task-to-CTA assignment."""

import json
from pathlib import Path

import torch
from layer_fixture import require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import (
    skinny_gemm,
)


def main():
    require_gpu0()
    rows = []
    for n, k in ((4608, 8192), (8192, 2048), (8192, 4096)):
        for seed in (7, 20261117):
            torch.manual_seed(seed)
            x = torch.randn((8, k), device="cuda:0", dtype=torch.bfloat16)
            w = torch.randn((n, k), device="cuda:0", dtype=torch.bfloat16) * (k**-0.5)
            expected = x.double() @ w.double().T
            baseline = skinny_gemm(x, w)
            for ctas in (1, 37, 192):
                out = skinny_gemm(x, w, ctas=ctas)
                torch.testing.assert_close(out.double(), expected, atol=0.02, rtol=0.02)
                assert torch.equal(out, baseline)
                diff = (out.double() - expected).abs()
                row = dict(
                    n=n,
                    k=k,
                    seed=seed,
                    ctas=ctas,
                    max_abs=float(diff.max()),
                    rel_linf=float(diff.max() / expected.abs().max()),
                    cta_assignment_bitwise=True,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    Path("/home/bangtliu/coord/logs/phase3b.0-core-validation.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(f"PASS {len(rows)} core cases", flush=True)


if __name__ == "__main__":
    main()

"""Fresh GPU0 process: _dense_partial is the first device compute kernel."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.gemm.bf16_splitk_gemm import _dense_partial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    assert (
        os.environ["HIP_VISIBLE_DEVICES"] == os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    )
    assert torch.cuda.device_count() == 1
    n = 18432 // args.tp
    gen = torch.Generator(device="cpu").manual_seed(7)
    x_cpu = torch.randn((8, 8192), generator=gen).to(torch.bfloat16)
    w_cpu = (torch.randn((n, 8192), generator=gen) / math.sqrt(8192)).to(torch.bfloat16)
    # Only CPU work and H2D copies precede the target kernel; no GPU random/fill/matmul.
    x, w = x_cpu.to("cuda:0"), w_cpu.to("cuda:0")
    out = torch.empty((8, 8, n), device="cuda:0", dtype=torch.float32)
    config = dict(
        M=8,
        N=n,
        K=8192,
        BM=16,
        BN=32,
        BK=128,
        SK=8,
        CACHE=".cg",
        num_warps=4,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    compiled = _dense_partial.warmup(x, w, out, **config, grid=(n // 32, 1, 8))
    prefix = args.output_prefix or Path(
        f"/home/bangtliu/coord/logs/phase3-S1x-tp{args.tp}-sum"
    )
    Path(str(prefix) + ".amdgcn").write_text(compiled.asm["amdgcn"])
    print(
        json.dumps(
            dict(
                event="launch",
                tp=args.tp,
                first_device_compute=True,
                torch=torch.__version__,
                triton=triton.__version__,
                config=config,
            )
        ),
        flush=True,
    )
    _dense_partial[(n // 32, 1, 8)](x, w, out, **config)
    torch.cuda.synchronize(0)
    actual = out.cpu().double()
    expected = torch.stack(
        [
            x_cpu[:, i * 1024 : (i + 1) * 1024].double()
            @ w_cpu[:, i * 1024 : (i + 1) * 1024].double().T
            for i in range(8)
        ]
    )
    finite = torch.isfinite(actual)
    difference = (actual - expected).abs()
    result = dict(
        event="result",
        tp=args.tp,
        nonfinite=int((~finite).sum()),
        total=actual.numel(),
        finite_max_abs=float(difference[finite].max()) if finite.any() else None,
        correct=bool(torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)),
    )
    full_reference = x_cpu.double() @ w_cpu.double().T
    summed = actual.sum(dim=0)
    finite_sum = torch.isfinite(summed)
    result["sum_nonfinite"] = int((~finite_sum).sum())
    result["sum_max_abs_finite"] = (
        float((summed - full_reference).abs()[finite_sum].max())
        if finite_sum.any()
        else None
    )
    result["sum_correct"] = bool(
        torch.allclose(summed, full_reference, rtol=1e-4, atol=1e-5)
    )
    print(json.dumps(result), flush=True)
    Path(str(prefix) + ".json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Bounded GPU0 skinny-GEMM probes with same-byte streaming comparisons."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import triton
from bench_hbm import graph_times, stream_read
from layer_fixture import require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import (
    _standalone,
)

LOG = Path("/home/bangtliu/coord/logs")


def run_shape(n, k, *, sweep):
    torch.manual_seed(7)
    x = torch.randn((8, k), device="cuda:0", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda:0", dtype=torch.bfloat16) * (k**-0.5)
    expected = x.double() @ w.double().T
    source = torch.rand((n * k // 2,), device="cuda:0", dtype=torch.float32)
    scratch = torch.empty((triton.cdiv(source.numel(), 4096),), device="cuda:0")
    stream = lambda: stream_read[(scratch.numel(),)](
        source, scratch, source.numel(), 4096
    )
    stream()
    torch.testing.assert_close(scratch, source.reshape(-1, 4096).sum(1))
    read_time = graph_times(stream)
    from aiter.tuned_gemm import tgemm

    native = lambda: tgemm.mm(x, w, None, otype=torch.bfloat16)
    native_time = graph_times(native)
    print(
        json.dumps(
            dict(
                event="baseline",
                n=n,
                k=k,
                bytes=n * k * 2,
                read=read_time,
                sglang_projection=native_time,
            )
        ),
        flush=True,
    )
    configs = [dict(waves=8, bn=16, bk=128, vec=8, stages=2, cache=".cs", direct=False)]
    if sweep:
        for change in [
            dict(waves=4, bk=256),
            dict(waves=16, bk=64),
            dict(vec=4),
            dict(stages=1),
            dict(stages=3),
            dict(cache=""),
            dict(cache=".cg"),
            dict(direct=True),
            dict(bn=32),
            dict(bn=32, bk=256),
            dict(bn=32, waves=16, bk=64),
            dict(bn=32, waves=16, bk=64, stages=3),
            dict(bn=64, waves=16, bk=64),
            dict(bn=64, waves=16, bk=128),
            dict(bn=64, waves=16, bk=256, stages=1),
            dict(bn=64, bk=512, stages=1),
        ]:
            configs.append(dict(configs[0], **change))
    if not sweep:
        configs = [
            dict(
                waves=16,
                bn=32 if n == 4608 else 64,
                bk=64 if n == 4608 else 128,
                vec=8,
                stages=2,
                cache=".cs",
                direct=False,
            )
        ]
    rows = []
    for i, c in enumerate(configs):
        if k // (c["waves"] // (c["bn"] // 16)) // c["bk"] < c["stages"]:
            continue
        out = torch.empty((8, n), device="cuda:0", dtype=torch.bfloat16)
        fn = lambda: _standalone[(192,)](
            x,
            w,
            out,
            n,
            k,
            c["bn"],
            c["bk"],
            c["vec"],
            c["stages"],
            c["cache"],
            c["direct"],
            num_warps=c["waves"],
        )
        compiled = fn()
        torch.cuda.synchronize(0)
        try:
            torch.testing.assert_close(out.double(), expected, atol=0.02, rtol=0.02)
        except AssertionError as exc:
            print(
                json.dumps(dict(event="rejected", n=n, k=k, config=c, reason=str(exc))),
                flush=True,
            )
            continue
        error = (out.double() - expected).abs()
        time = graph_times(fn)
        asm = compiled.asm["amdgcn"]
        digest = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]
        asm_path = LOG / f"phase3b.0-n{n}-k{k}-cfg{digest}.amdgcn"
        asm_path.write_text(asm)
        private = re.search(r"\.amdhsa_private_segment_fixed_size\s+(\d+)", asm)
        row = dict(
            event="candidate",
            asm_path=str(asm_path),
            n=n,
            k=k,
            config=c,
            **time,
            max_abs=float(error.max()),
            rel_linf=float(error.max() / expected.abs().max()),
            read_median_us=read_time["median_us"],
            native_median_us=native_time["median_us"],
            read_ratio=time["median_us"] / read_time["median_us"],
            b128_load_instructions=asm.count("global_load_b128"),
            private_segment_bytes=int(private.group(1)) if private else None,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        (LOG / f"phase3b.0-n{n}-k{k}-results.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
    (LOG / f"phase3b.0-n{n}-k{k}-results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--all-shapes", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shape", choices=["qkvg", "o", "unpool"], default="qkvg")
    args = parser.parse_args()
    require_gpu0()
    for n, k in (
        [(4608, 8192), (8192, 2048), (8192, 4096)]
        if args.all_shapes
        else [dict(qkvg=(4608, 8192), o=(8192, 2048), unpool=(8192, 4096))[args.shape]]
    ):
        run_shape(n, k, sweep=args.sweep)


if __name__ == "__main__":
    main()

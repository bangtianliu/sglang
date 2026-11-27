"""T10 size/grid-matched stream ceilings and an in-flight-byte upper bound."""

import argparse
import json
from pathlib import Path

import torch
import triton
from bench_tlx_ceiling import stream
from layer_fixture import require_gpu0
from paired45 import measure
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--mib", type=float, required=True)
    p.add_argument("--label", default="phaseT10")
    a = p.parse_args()
    size = int(a.mib * (1 << 20))
    assert size > 0 and size % 16384 == 0
    n = size // 4
    blocks = triton.cdiv(n, 4096)
    xs = [torch.ones(n, device="cuda") for _ in range(3)]
    rows = []
    for grid in (192, 288, 512, 768, 1728):
        cases = {}
        resources = {}
        for name, g in [("resident", grid), ("wide", blocks)]:
            ys = [torch.empty(g, device="cuda") for _ in xs]
            ids = torch.arange(g, device="cuda", dtype=torch.int64)
            expected = torch.clamp((blocks - 1 - ids) // g + 1, min=0).float() * 4096

            def call(g=g, ys=ys):
                for x, y in zip(xs, ys):
                    stream[(g,)](x, y, n, g, "", num_warps=4)

            def check(ys=ys, expected=expected):
                for copy, y in enumerate(ys):
                    bad = (y != expected).nonzero()
                    assert len(bad) == 0, (
                        "stream",
                        "rank0",
                        copy,
                        bad[0].tolist() if len(bad) else None,
                    )

            kernel = stream.warmup(xs[0], ys[0], n, g, "", num_warps=4, grid=(g,))
            resource = kernel_occupancy("size-stream", kernel)
            resources[name] = dict(
                grid=g,
                resources=dump(
                    kernel, f"{a.label}-size-stream-{a.mib:g}/{grid}-{name}", "tlx"
                ),
                resident_ctas_per_cu=resource.active_ctas_per_cu,
                logical_block_bytes=4096 * 4,
                inflight_bytes_per_cu_upper=[
                    min(g // 192, resource.active_ctas_per_cu) * 16384,
                    min(triton.cdiv(g, 192), resource.active_ctas_per_cu) * 16384,
                ],
            )
            cases[name] = (lambda: None, call, check)
        result, _ = measure(cases)
        for name, data in result.items():
            data["per_copy_us"] = data["median_us"] / 3
            data["logical_TB_s"] = size / data["per_copy_us"] / 1e6
            data["needed_inflight_bytes_per_cu_at_363ns"] = (
                data["logical_TB_s"] * 1e12 * 363e-9 / 192
            )
        rows.append(dict(grid=grid, results=result, admissions=resources))
        print(
            a.mib,
            grid,
            {name: data["logical_TB_s"] for name, data in result.items()},
            flush=True,
        )
    output = dict(
        mib=a.mib,
        bytes=size,
        copies=3,
        rows=rows,
        note="Logical bytes/time, not physical HBM counters. In-flight bound assumes a complete16KiBload block outstanding per resident CTA; ISA wait placement can make effective bytes lower. Little law estimate uses the independent T9cold dependent-read363ns as a model, not measured stream-request latency. All sizes are multiples of4096floats; checksum exact.",
    )
    Path("/home/bangtliu/coord/logs", f"{a.label}-stream-{a.mib:g}.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

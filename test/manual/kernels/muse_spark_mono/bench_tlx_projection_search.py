"""GPU0 T10 projection load search, cold three-copy paired AB/BA."""

import argparse
import functools
import hashlib
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import torch
from bench_tlx_decomposition import Capture, setup
from layer_fixture import require_gpu0
from paired45 import measure
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_chain as chain,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_projection as projection,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_projection_diagnostic as reference,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_projection_load_search as search,
)

LOG = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=("S1", "S4", "S6", "S3"), required=True)
    p.add_argument("--label", default="phaseT10-projection-search")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    retained, captures = [], []
    for _ in range(3):
        base, native, *_ = setup(8192, -1)
        x, stream, w, m, _ = base["chained"][1].args
        ws = chain.make_workspace("cuda", context=8192, left=-1)
        retained.append((base, native, ws))
        chain.prepare_peers(ws, *native.partials, rank=0)
        captured = {}
        with patch.object(
            projection, "project", Capture(projection.project, captured, "project")
        ):
            chain.launch(x, stream, w, m, ws, rank=0)
        torch.cuda.synchronize()
        grid, args, opts = captured[a.kernel]
        source = args[0].cpu().view(torch.int16).to(torch.int64) & 65535
        weights = args[1].cpu().view(torch.int16).to(torch.int64) & 65535
        captures.append((grid, args, opts, source, weights))
    n, k = captures[0][2]["N"], captures[0][2]["K"]
    candidates = []
    for ps, d, bk, copy in itertools.product(
        (1, 2, 4, 8), (1, 2, 4), (512, 1024), ("tdm", "async")
    ):
        if k // ps < bk or (k // ps) % bk or d > max(2, k // ps // bk):
            continue
        if a.quick and (bk != 512 or copy != "tdm" or d not in (1, 2)):
            continue
        candidates.append((ps, d, bk, copy))
    admissions = []

    def build(config):
        rr = []
        for index, (grid, args, opts, source, weights) in enumerate(captures):
            if config is None:
                ps, d, bk, copy = 1, 4, 512, "tdm"
                output = torch.empty((n // 16,), device="cuda", dtype=torch.uint32)
                status = args[3]
                options = dict(opts, MODE="load", Checksum=output)
                fn = reference.project
                aa = args
            else:
                ps, d, bk, copy = config
                grid = (n // 16 * ps,)
                output = torch.empty(grid, device="cuda", dtype=torch.uint32)
                status = torch.empty(grid, device="cuda", dtype=torch.int32)
                fn = search.load_project
                aa = (args[0], args[1], output, status)
                options = dict(
                    N=n,
                    K=k,
                    PSPLIT=ps,
                    DEPTH=d,
                    BK=bk,
                    COPY=copy,
                    STAGE=int(a.kernel[1:]),
                    Failure=opts.get("Failure"),
                    num_warps=2,
                    enable_fp_fusion=False,
                )
            ks = (
                torch.arange(ps)[:, None] * (k // ps)
                + torch.arange(k // ps // bk)[None, :] * bk
            )
            expected = (
                source[0, ks]
                + source[0, ks + 1]
                + weights[::16][:, ks]
                + weights[::16][:, ks + 1]
            ).sum(-1).reshape(-1) & 0xFFFFFFFF
            kernel = fn.warmup(*aa, **options, grid=grid)
            r = kernel_occupancy(a.kernel + "-load-search", kernel)
            assert kernel.n_spills == 0 and r.active_ctas_per_cu >= 1, (
                a.kernel,
                config,
                r,
                kernel.n_spills,
            )
            name = "reference" if config is None else f"p{ps}-d{d}-k{bk}-{copy}"
            if index == 0:
                admissions.append(
                    dict(
                        config=config,
                        grid=grid,
                        resources=dump(
                            kernel, a.label + "/" + a.kernel + "/" + name, "tlx"
                        ),
                        occupancy=r.active_ctas_per_cu,
                    )
                )

            def call(fn=fn, grid=grid, aa=aa, options=options):
                return fn[grid](*aa, **options)

            def prepare(output=output, status=status):
                output.fill_(0xFFFFFFFF)
                status.zero_()

            def check(
                output=output, status=status, expected=expected, index=index, name=name
            ):
                got = output.cpu().to(torch.int64)
                bad = (got != expected).nonzero()
                loc = int(bad[0, 0]) if len(bad) else None
                assert not len(bad), (
                    a.kernel,
                    name,
                    "rank0",
                    "checksum",
                    index,
                    loc,
                    int(got[loc]) if loc is not None else None,
                    int(expected[loc]) if loc is not None else None,
                )
                bad = (status != 1).nonzero()
                assert not len(bad), (
                    a.kernel,
                    name,
                    "rank0",
                    "status",
                    index,
                    bad[:1].tolist(),
                )

            rr.append((prepare, call, check))

        def action(i):
            for row in rr:
                row[i]()

        return tuple(functools.partial(action, i) for i in range(3))

    base = build(None)
    results = {}
    for config in candidates:
        name = f"p{config[0]}-d{config[1]}-k{config[2]}-{config[3]}"
        pair, _ = measure({"reference": base, "candidate": build(config)})
        per_copy = {
            side: {
                order: data["by_order"][order]["median_us"] / 3
                for order in ("AB", "BA")
            }
            for side, data in pair.items()
        }
        results[name] = dict(config=config, pair=pair, per_copy_us=per_copy)
        data = dict(
            kernel=a.kernel,
            N=n,
            K=k,
            copies=3,
            results=results,
            admissions=admissions,
            source_hashes={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(search.__file__))
            },
            notes="Load-only; complete original operand transactions, sampled two-word checksum per TDM tile. PSPLIT changes CTA ownership only. Canonical storage, same logical weight and activation bytes, no flags/peers/concurrent kernels.",
        )
        (LOG / f"{a.label}-{a.kernel}.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )
        print("PASS", a.kernel, name, per_copy, flush=True)


if __name__ == "__main__":
    main()

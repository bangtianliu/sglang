"""T10 GPU0 same-body full/load/compute decomposition, cold three-copy pairs.

Diagnostic modules are never selected by the production chain. Global operand
transactions and dynamic math counts are separately checked in CPU IR proofs.
"""

import argparse
import functools
import json
from pathlib import Path
from unittest.mock import patch

import torch
from bench_chained47 import setup
from check_chained47 import archive_check
from layer_fixture import require_gpu0
from layer_reference import exact_check
from paired45 import measure
from reference_archive import gpu
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_chain as chain,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_expert_diagnostic as expert_diag,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_experts as experts,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_projection as projection,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_projection_diagnostic as projection_diag,
)

LOG = Path("/home/bangtliu/coord/logs")


class Capture:
    def __init__(self, fn, rows, kind):
        self.fn, self.rows, self.kind = fn, rows, kind

    def __getattr__(self, name):
        return getattr(self.fn, name)

    def __getitem__(self, grid):
        invoke = self.fn[grid]

        def call(*aa, **kw):
            name = (
                ("W13" if kw["FIRST"] else "W2")
                if self.kind == "expert"
                else "S" + str(kw["STAGE"])
            )
            self.rows[name] = (grid, aa, dict(kw))
            return invoke(*aa, **kw)

        return call


def word_sum(value):
    words = value.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return int(words.sum()) & 0xFFFFFFFF


def expected_expert(aa, first, task):
    x, sorted_ids, expert_ids, _, w, scale, *_ = aa
    block = task // 64
    tile = task % 64
    expert = int(expert_ids[block].cpu())
    slots = sorted_ids[block * 16 : block * 16 + 16].cpu().long()
    active = slots < 64
    source = x.cpu()
    a = torch.zeros((16, 4096), dtype=torch.bfloat16)
    a[active] = source[slots[active] // 8 if first else slots[active]]
    answer = word_sum(a)
    w = w.reshape(64, 8192 if first else 4096, 2048)
    scale = scale.reshape(64, 8192 if first else 4096, 128)
    for offset in (0, 4096) if first else (0,):
        lo = tile * 64 + offset
        answer += word_sum(w[expert, lo : lo + 64].cpu())
        answer += 128 * word_sum(scale[expert, lo : lo + 64].cpu())
    return answer & 0xFFFFFFFF


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--kernel", required=True, choices=("S1", "S3", "S4", "S6", "W13", "W2")
    )
    p.add_argument("--label", default="phaseT10-decomposition")
    p.add_argument("--wide", action="store_true")
    a = p.parse_args()
    if a.wide:
        assert a.kernel in ("W13", "W2")
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
            tlx_expert_wide_diagnostic,
        )

    is_expert = a.kernel.startswith("W")
    first = a.kernel == "W13"
    retained = []
    records = []
    rows = {n: [] for n in ("full", "load", "compute")}
    admissions = []
    traffic = []
    for copy in range(3):
        base, native, _, _, _ = setup(8192, -1)
        retained.append((base, native))
        x, stream, w, m, _ = base["chained"][1].args
        ws = chain.make_workspace("cuda", context=8192, left=-1)
        retained.append(ws)
        captured = {}
        chain.prepare_peers(ws, *native.partials, rank=0)
        with (
            patch.object(
                projection, "project", Capture(projection.project, captured, "project")
            ),
            patch.object(
                experts,
                "expert_phase",
                Capture(experts.expert_phase, captured, "expert"),
            ),
        ):
            chain.launch(x, stream, w, m, ws, rank=0)
        torch.cuda.synchronize()
        old = gpu(
            torch.load(
                "/home/bangtliu/coord/phase44-baseline/layer-7-8192--1.pt",
                weights_only=True,
            )["rows"][0]
        )
        archive_check(native.f, ws, old, 0, digest=False)
        grid, aa, kw = captured[a.kernel]
        reference = experts.expert_phase if is_expert else projection.project
        fn = expert_diag.expert_phase if is_expert else projection_diag.project
        if a.wide:
            fn = tlx_expert_wide_diagnostic.expert_phase
        reference[grid](*aa, **kw)
        torch.cuda.synchronize()
        original = aa[6 if is_expert else 2].clone()
        if is_expert:
            tasks = int(aa[3].cpu()) // 16 * 64
            slots = aa[1][: tasks // 64 * 16].reshape(-1, 16)
            hits = tuple(int(v) for v in (slots < 64).sum(1).cpu())
            selected = sorted(set((0, 63, (tasks // 128) * 64 + 17, tasks - 1)))
            expected = {task: expected_expert(aa, first, task) for task in selected}
            active_experts = int(torch.unique(aa[2][: tasks // 64]).numel())
            weights_scales = active_experts * (8192 if first else 4096) * (2048 + 128)
            metadata = dict(
                tasks=tasks,
                hits=hits,
                active_experts=active_experts,
                weight_scale_bytes=weights_scales,
            )
        else:
            n, k = kw["N"], kw["K"]
            tasks = n // 16
            source = aa[0].cpu().view(torch.int16).to(torch.int64) & 65535
            weight = aa[1].cpu().view(torch.int16).to(torch.int64) & 65535
            ks = torch.arange(k // 512) * 512
            expected = (
                source[0, ks].sum()
                + source[0, ks + 1].sum()
                + weight[::16, ks].sum(1)
                + weight[::16, ks + 1].sum(1)
            ) & 0xFFFFFFFF
            metadata = dict(
                tasks=tasks,
                weight_scale_bytes=n * k * 2,
                logical_activation_bytes=8 * k * 2 * tasks,
            )
        traffic.append(metadata)
        for mode in rows:
            args = list(aa)
            opts = dict(kw, MODE=mode)
            opts.pop("WIDE", None)
            if a.wide:
                opts.update(
                    PACK_W=False,
                    PACK_S=False,
                    TDM=False,
                    DEPTH=2,
                    PREFETCH_FIRST=True,
                    COMPUTE_K=128,
                    SCALE_ASYNC=True,
                    waves_per_eu=2,
                )
            if mode == "load":
                checksum = torch.zeros(
                    (max(tasks, 1), 128) if is_expert else (tasks,),
                    device="cuda",
                    dtype=torch.uint32,
                )
                opts["Checksum"] = checksum
            else:
                opts["Checksum"] = None
            if is_expert:
                opts.update(
                    TASKS=tasks if mode == "compute" else 0,
                    HITS=hits if mode == "compute" else (),
                )
            if mode == "compute":
                if is_expert:
                    args[8] = None
                    args[6] = torch.empty(
                        (1024, 4096), device="cuda", dtype=torch.bfloat16
                    )
                else:
                    opts["Failure"] = None
                    args[2] = torch.empty_like(args[2])
                    if opts.get("Packed") is not None:
                        opts["Packed"] = torch.empty_like(opts["Packed"])
            target = reference if a.wide and mode == "full" else fn
            if a.wide and mode == "full":
                opts = dict(kw)
            kernel = target.warmup(*args, **opts, grid=grid)
            resource = kernel_occupancy(a.kernel + "-" + mode, kernel)
            assert resource.active_ctas_per_cu >= 1 and kernel.n_spills == 0, (
                a.kernel,
                mode,
                resource,
                kernel.n_spills,
            )
            if copy == 0:
                admissions.append(
                    dict(
                        mode=mode,
                        grid=list(grid),
                        resources=dump(
                            kernel, f"{a.label}-GPU-{a.kernel}/{mode}", "tlx"
                        ),
                        active_ctas_per_cu=resource.active_ctas_per_cu,
                    )
                )

            def call(args=tuple(args), opts=opts, target=target):
                return target[grid](*args, **opts)

            def check(
                mode=mode,
                args=args,
                checksum=checksum if mode == "load" else None,
                original=original,
                expected=expected,
                hits=hits if is_expert else (),
                copy=copy,
            ):
                if mode == "full":
                    actual = args[6 if is_expert else 2]
                    exact_check(
                        actual.view(torch.uint8),
                        original.view(torch.uint8),
                        a.kernel + " full output bytes copy" + str(copy),
                        {"_rank": 0},
                    )
                elif mode == "load":
                    if is_expert:
                        for task, want in expected.items():
                            got = (
                                int(checksum[task].cpu().to(torch.int64).sum())
                                & 0xFFFFFFFF
                            )
                            assert got == want, (
                                a.kernel,
                                "load checksum",
                                "rank0",
                                copy,
                                task,
                                got,
                                want,
                            )
                    else:
                        exact_check(
                            checksum.cpu().to(torch.int64),
                            expected,
                            a.kernel + " load checksum copy" + str(copy),
                            {"_rank": 0},
                        )
                else:
                    actual = args[6] if is_expert else args[2].view(torch.float32)
                    if is_expert:
                        valid = [
                            block * 16 + row
                            for block, count in enumerate(hits)
                            for row in range(count)
                        ]
                        actual = actual[valid]
                    bad = (~torch.isfinite(actual)).nonzero()
                    assert len(bad) == 0, (
                        a.kernel,
                        "compute finite",
                        "rank0",
                        copy,
                        bad[0].tolist() if len(bad) else None,
                    )
                exact_check(
                    args[7 if is_expert else 3],
                    1,
                    a.kernel + " status " + mode + " copy" + str(copy),
                    {"_rank": 0},
                )

            rows[mode].append((lambda: None, call, check))
    cases = {}
    for mode, rr in rows.items():

        def action(index, rr=rr):
            for row in rr:
                row[index]()

        cases[mode] = tuple(functools.partial(action, i) for i in range(3))
    pairs = {}
    for mode in ("load", "compute"):
        result, _ = measure({n: cases[n] for n in ("full", mode)})
        result["per_copy_us"] = {
            n: {order: r["by_order"][order]["median_us"] / 3 for order in ("AB", "BA")}
            for n, r in result.items()
        }
        pairs[mode] = result
    data = dict(
        kernel=a.kernel,
        wide=a.wide,
        copies=3,
        context=8192,
        left=-1,
        traffic=traffic,
        admissions=admissions,
        pairs=pairs,
        notes="All modes same grid and original pipeline shape. Full exact vs production. Projection load mode samples two BF16 words per complete TDM tile (all descriptor transactions retained in CPU IR). Expert load mode checksums all copied bytes, including scales. Compute uses representative LDS initialization (included in time), specialized task/hit control, no global data loads or spills. Per-copy times divide three rotating-copy graph totals by3. Pairwise same-process AB/BA,100samples/frontend/order; no kernel flag wait or physical peer access.",
    )
    (LOG / f"{a.label}-{a.kernel}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        "PASS",
        a.kernel,
        {mode: r["per_copy_us"] for mode, r in pairs.items()},
        flush=True,
    )


if __name__ == "__main__":
    main()

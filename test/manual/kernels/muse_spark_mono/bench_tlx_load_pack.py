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


def expected_expert(aa, first, task, bn=64):
    x, sorted_ids, expert_ids, _, w, scale, *_ = aa
    block = task // (4096 // bn)
    tile = task % (4096 // bn)
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
        lo = tile * bn + offset
        answer += word_sum(w[expert, lo : lo + bn].cpu())
        answer += 128 * word_sum(scale[expert, lo : lo + bn].cpu())
    return answer & 0xFFFFFFFF


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True, choices=("W13", "W2"))
    p.add_argument("--label", default="phaseT10-load-pack")
    p.add_argument("--scale-async", action="store_true")
    p.add_argument("--prefetch-first", action="store_true")
    p.add_argument("--grid", type=int, default=1728)
    p.add_argument("--depth", type=int, default=2, choices=(2, 3, 4, 5, 6, 8))
    p.add_argument("--tdm", action="store_true")
    p.add_argument("--bk", type=int, default=128, choices=(128, 256, 512, 1024))
    p.add_argument("--waves-per-eu", type=int, default=9)
    p.add_argument("--cache", default="", choices=("", ".cs"))
    p.add_argument("--reference-best", action="store_true")
    p.add_argument("--reference-path")
    p.add_argument("--canonical-only", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--reference-bk", type=int, default=0)
    p.add_argument("--reference-waves", type=int, default=-1)
    p.add_argument("--compute-k", type=int, default=128, choices=(128, 256))
    p.add_argument("--bn", type=int, default=64, choices=(32, 64))
    p.add_argument("--num-warps", type=int, default=4)
    p.add_argument("--apad", action="store_true")
    a = p.parse_args()
    reference_module = None
    if a.reference_path:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "T10_load_reference", a.reference_path
        )
        reference_module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = reference_module
        spec.loader.exec_module(reference_module)
    if a.bk != 128:
        import importlib

        global expert_diag
        expert_diag = importlib.import_module(
            "sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_load_k"
            + str(a.bk)
        )
    if a.apad:
        assert a.bn == 64 and a.bk == 512
        expert_diag = importlib.import_module(
            "sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_load_apad"
        )
    if a.bn == 32:
        assert (
            a.kernel == "W2"
            and a.bk == 512
            and a.canonical_only
            and not a.reference_best
            and not a.tdm
        )
        expert_diag = importlib.import_module(
            "sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_load_n32"
        )
    if a.tdm and a.bk == 128:
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
            tlx_expert_load_tdm as expert_diag,
        )
    assert a.prefetch_first or a.depth == 2
    is_expert = a.kernel.startswith("W")
    first = a.kernel == "W13"
    pflags = {
        "raw": (False, False),
        "pipeline": (False, False),
        "weight": (True, False),
        "scale": (False, True),
        "both": (True, True),
    }
    if a.canonical_only:
        pflags = {n: pflags[n] for n in ("raw", "pipeline")}
    if a.reference_best:
        pflags["raw"] = (True, True)
    retained = []
    records = []
    rows = {n: [] for n in pflags}
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
            from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
                tlx_expert_load_search as base128,
            )

            fn = (
                base128.expert_phase
                if mode == "raw" and not a.reference_best
                else expert_diag.expert_phase
            )
            if mode == "raw" and reference_module is not None:
                fn = reference_module.expert_phase
            local_bn = (
                64
                if mode == "raw" and reference_module is None and not a.reference_best
                else a.bn
            )
            tiles = 4096 // local_bn
            tasks = int(aa[3].cpu()) // 16 * tiles
            selected = sorted(
                set((0, tiles - 1, (tasks // (2 * tiles)) * tiles + 17, tasks - 1))
            )
            expected = {
                task: expected_expert(aa, first, task, local_bn) for task in selected
            }
            args = list(aa)
            pw, ps = pflags[mode]
            opts = dict(kw, MODE="full" if a.full else "load", PACK_W=pw, PACK_S=ps)
            opts.pop("WIDE", None)
            launch_grid = (
                grid
                if mode == "raw" and not a.reference_best and reference_module is None
                else (a.grid,)
            )
            if mode != "raw" or a.reference_best or reference_module is not None:
                opts.update(
                    SCALE_ASYNC=a.scale_async,
                    PREFETCH_FIRST=a.prefetch_first,
                    GRID=a.grid,
                    DEPTH=a.depth,
                    waves_per_eu=a.waves_per_eu,
                    num_warps=a.num_warps,
                    WEIGHT_CACHE="" if mode == "raw" else a.cache,
                )
            if mode == "raw" and a.reference_waves >= 0:
                opts["waves_per_eu"] = a.reference_waves
            if mode != "raw" and a.bk == 512 and a.bn == 64:
                opts["COMPUTE_K"] = a.compute_k
            if a.tdm and mode != "raw":
                opts["TDM"] = True
            for idx, packed in ((4, pw), (5, ps)):
                if packed:
                    value = args[idx]
                    shape = value.shape
                    n = 8192 if first else 4096
                    width = 2048 if idx == 4 else 128
                    pack_bk = (
                        a.reference_bk if mode == "raw" and a.reference_bk else a.bk
                    )
                    chunk = pack_bk // 2 if idx == 4 else pack_bk // 32
                    view = value.reshape(64, n // 64, 64, 4096 // pack_bk, chunk)
                    result = view.permute(0, 1, 3, 2, 4).contiguous()
                    assert torch.equal(
                        result.permute(0, 1, 3, 2, 4).contiguous(), view
                    ), ("offline permutation mapping", idx, copy)
                    args[idx] = result.reshape(shape)
                    retained.append(args[idx])
            checksum = None
            if mode in pflags and not a.full:
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
            kernel = fn.warmup(*args, **opts, grid=launch_grid)
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
                        bn=local_bn,
                        tasks=tasks,
                        logical_activation_bytes=sum(hits)
                        * 4096
                        * 2
                        * (4096 // local_bn),
                        grid=list(launch_grid),
                        resources=dump(kernel, f"{a.label}-{a.kernel}/{mode}", "tlx"),
                        active_ctas_per_cu=resource.active_ctas_per_cu,
                    )
                )

            def call(args=tuple(args), opts=opts, launch_grid=launch_grid, fn=fn):
                return fn[launch_grid](*args, **opts)

            def check(
                mode=mode,
                args=args,
                checksum=checksum,
                original=original,
                expected=expected,
                hits=hits if is_expert else (),
                copy=copy,
            ):
                if a.full:
                    actual = args[6 if is_expert else 2]
                    exact_check(
                        actual.view(torch.uint8),
                        original.view(torch.uint8),
                        a.kernel + " full output bytes copy" + str(copy),
                        {"_rank": 0},
                    )
                elif mode in pflags:
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
                                mode,
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

            if a.full:
                active_rows = aa[1][: tasks // 64 * 16]
                active_rows = active_rows[active_rows < 64].long()

                def prepare(args=args, active_rows=active_rows):
                    args[6].index_fill_(0, active_rows, float("nan"))
                    args[7].zero_()
            else:
                prepare = lambda: None
            rows[mode].append((prepare, call, check))
    cases = {}
    for mode, rr in rows.items():

        def action(index, rr=rr):
            for row in rr:
                row[index]()

        cases[mode] = tuple(functools.partial(action, i) for i in range(3))
    pairs = {}
    for mode in (name for name in rows if name != "raw"):
        result, _ = measure({n: cases[n] for n in ("raw", mode)})
        result["per_copy_us"] = {
            n: {order: r["by_order"][order]["median_us"] / 3 for order in ("AB", "BA")}
            for n, r in result.items()
        }
        pairs[mode] = result
    import hashlib

    source_files = [Path(__file__), Path(expert_diag.__file__), Path(base128.__file__)]
    if a.reference_path:
        source_files.append(Path(a.reference_path))
    source_hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
    }
    data = dict(
        source_hashes=source_hashes,
        search_options=vars(a),
        memory_cost="Replacement format no padding, unchanged bytes. Keeping both canonical+packed adds 1.59375 GiB/rank/layer (95.625 GiB/60 layers); scale-only adds96 MiB/layer. Allocation/permutation outside timing.",
        kernel=a.kernel,
        copies=3,
        context=8192,
        left=-1,
        traffic=traffic,
        admissions=admissions,
        pairs=pairs,
        notes="Load-only all copied bytes checksum; no WMMA. Exact inverse startup permutation checked. Same original tile/grid/pipeline; only offline weight and/or scale addresses differ. Three distinct rotating weights, pairwise same-process AB/BA; per-copy time divides graph of3 by3. No flags or peer access.",
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

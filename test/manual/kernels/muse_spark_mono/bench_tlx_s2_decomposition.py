"""GPU0 S2 partial load/math/full decomposition with cold rotating KV pools."""

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
    tlx_attention as stage,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_attention_diagnostic as diag,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_chain as chain,
)

LOG = Path("/home/bangtliu/coord/logs")


class Capture:
    def __init__(self, fn, rows):
        self.fn, self.rows = fn, rows

    def __getattr__(self, key):
        return getattr(self.fn, key)

    def __getitem__(self, grid):
        def call(*aa, **kw):
            self.rows.append((grid, aa, kw))
            return self.fn[grid](*aa, **kw)

        return call


def expected(aa, kw, pid):
    _, k, v, pages, mapping, length, *_ = aa
    length = int(length.cpu())
    head = pid % 4
    split = pid // 4
    first = max(0, length - 8 - kw["LEFT"]) // 64 * 64 if kw["LEFT"] >= 0 else 0
    tiles = (length - first + 63) // 64
    start = first + split * tiles // kw["SPLITS"] * 64
    end = min(length, first + (split + 1) * tiles // kw["SPLITS"] * 64)
    pages = pages.cpu().reshape(-1)
    mapping = mapping.cpu().reshape(-1) if mapping is not None else None
    k = k.cpu().view(torch.uint8).reshape(-1, 4, 64)
    v = v.cpu().view(torch.uint8).reshape(-1, 4, 64)
    total = 0
    for offset in range(start, end, 64):
        page = int(pages[offset // 16])
        slot = page * 16 + offset % 16
        if kw["HAS_MAP"]:
            slot = (
                int(mapping[page * 16]) + offset % 16
                if kw["MAP_PAGE"]
                else int(mapping[slot])
            )
        if slot >= 0:
            total += int(
                k[slot, head, :16].to(torch.int64).sum()
                + v[slot, head, :16].to(torch.int64).sum()
            )
    return total & 0xFFFFFFFF


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--label", default="phaseT11-S2-decomposition")
    a = p.parse_args()
    retained = []
    rows = {m: [] for m in ("full", "load", "compute")}
    admissions = []
    pointers = []
    traffic = []
    for copy in range(3):
        base, native, _, _, _ = setup(a.context, a.left)
        retained.append((base, native))
        x, stream, w, m, _ = base["chained"][1].args
        ws = chain.make_workspace("cuda", context=a.context, left=a.left)
        retained.append(ws)
        captured = []
        chain.prepare_peers(ws, *native.partials, rank=0)
        with patch.object(stage, "partials", Capture(stage.partials, captured)):
            chain.launch(x, stream, w, m, ws, rank=0)
        torch.cuda.synchronize()
        old = gpu(
            torch.load(
                LOG.parent / "phase44-baseline" / f"layer-7-{a.context}-{a.left}.pt",
                weights_only=True,
            )["rows"][0]
        )
        archive_check(native.f, ws, old, 0, digest=False)
        grid, aa, kw = captured[-1]
        length = int(aa[5].cpu())
        original = (aa[8].clone(), aa[9].clone())
        pointers.append([aa[1].data_ptr(), aa[2].data_ptr()])
        selected = sorted(set((0, 3, grid[0] // 2, grid[0] - 1)))
        wanted = {pid: expected(aa, kw, pid) for pid in selected}
        first = max(0, length - 8 - a.left) // 64 * 64 if a.left >= 0 else 0
        traffic.append(
            dict(
                length=length,
                splits=kw["SPLITS"],
                grid=grid,
                logical_KV_bytes=((length - first + 63) // 64) * 64 * 4 * 64 * 2,
                repeated_Q_bytes=grid[0] * 64 * 64 * 2,
            )
        )
        # Full diagnostic math must also reproduce production before timing it.
        diag.partials[grid](*aa, **kw, MODE="full", LENGTH=length, Checksum=None)
        torch.cuda.synchronize()
        for name, actual, ref in zip(("partial", "lse"), aa[8:10], original):
            exact_check(actual, ref, "S2 diagnostic full " + name, {"_rank": 0})
        for mode in rows:
            args = list(aa)
            opts = dict(kw)
            fn = stage.partials if mode == "full" else diag.partials
            checksum = (
                torch.empty((grid[0],), device="cuda", dtype=torch.uint32)
                if mode == "load"
                else None
            )
            if mode != "full":
                opts.update(MODE=mode, LENGTH=length, Checksum=checksum)
            if mode == "compute":
                args[8] = torch.empty_like(args[8])
                args[9] = torch.empty_like(args[9])
                args[12] = None
            kernel = fn.warmup(*args, **opts, grid=grid)
            resource = kernel_occupancy("S2-" + mode, kernel)
            assert kernel.n_spills == 0, (mode, kernel.n_spills)
            if copy == 0:
                admissions.append(
                    dict(
                        mode=mode,
                        resources=dump(
                            kernel, f"{a.label}-{a.context}-{a.left}/{mode}", "tlx"
                        ),
                        occupancy=resource.active_ctas_per_cu,
                    )
                )

            def call(args=tuple(args), opts=opts, fn=fn, grid=grid):
                return fn[grid](*args, **opts)

            def check(
                args=args,
                mode=mode,
                checksum=checksum,
                wanted=wanted,
                original=original,
                copy=copy,
            ):
                if mode == "full":
                    for name, actual, ref in zip(
                        ("partial", "lse"), args[8:10], original
                    ):
                        exact_check(actual, ref, f"S2 {name} copy{copy}", {"_rank": 0})
                elif mode == "load":
                    for pid, want in wanted.items():
                        assert int(checksum[pid].cpu().to(torch.int64)) == want, (
                            "S2 checksum",
                            copy,
                            pid,
                            int(checksum[pid]),
                            want,
                        )
                else:
                    actual = args[8].view(torch.float32)
                    bad = (~torch.isfinite(actual)).nonzero()
                    assert len(bad) == 0, (
                        "S2 compute finite",
                        copy,
                        bad[0].tolist() if len(bad) else None,
                    )
                exact_check(args[11], 1, f"S2 {mode} status copy{copy}", {"_rank": 0})

            rows[mode].append((lambda: None, call, check))
    assert len({r[0] for r in pointers}) == 3 and len({r[1] for r in pointers}) == 3
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
            n: {o: r["by_order"][o]["median_us"] / 3 for o in ("AB", "BA")}
            for n, r in result.items()
        }
        pairs[mode] = result
    result = dict(
        context=a.context,
        left=a.left,
        admissions=admissions,
        traffic=traffic,
        KV_pointers=pointers,
        pairs=pairs,
        notes="Load retains full FP8 async tiles, checksums first16bytes of row0 per K/V tile. Compute initializes both LDS banks (included), omits global operands/page tables/scales/epoch RMW. Full diagnostic equals production bits; timed full uses production entry. Three rotating distinct KV pools, pairwise100samples/order. No physical peers.",
    )
    (LOG / f"{a.label}-{a.context}-{a.left}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("PASS", {n: r["per_copy_us"] for n, r in pairs.items()}, flush=True)


if __name__ == "__main__":
    main()

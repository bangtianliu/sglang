"""ONE authorized T2 seed-7 case; guarded boundaries, no timing/adoption.

The projection descriptors and ring sizes are statically checked. Every explicit
lowered LLVM global/buffer/inline-asm/LDS access in both boundaries is guarded.
Compiler-generated private spill accesses are outside the IR instrumentation.
"""

import json
from pathlib import Path

import torch
import triton
from layer_fixture import require_gpu0
from prepare_t2_isolation import fault_module
from range_guard_t2 import guard_ir
from reference_archive import checked_load
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.runtime.cache import get_override_manager

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    independent_tail,
    tlx_projection,
    wide_projection,
)

ROOT = Path("/home/bangtliu/coord/logs/phaseT2-isolation")
NAMES = ("P", "Stream", "Post", "Pre", "Alpha", "Beta", "Out", "Updated", "Status")


def prepare(side, fn, args, opts):
    kernel = fn.warmup(*args, grid=(8,), **opts)
    (ROOT / (side + "-pre-guard.llir")).write_text(kernel.asm["llir"])
    bounds = {
        i: (t.data_ptr(), t.numel() * t.element_size()) for i, t in enumerate(args[:9])
    }
    _, sites = guard_ir(kernel.asm["llir"], bounds, kernel.metadata.shared, 0, 0)
    records = torch.zeros((len(sites), 1024, 4), device="cuda", dtype=torch.int64)
    dummy = torch.zeros((4096,), device="cuda", dtype=torch.uint8)
    guarded, sites = guard_ir(
        kernel.asm["llir"],
        bounds,
        kernel.metadata.shared,
        records.data_ptr(),
        dummy.data_ptr(),
    )
    folder = ROOT / side
    folder.mkdir(exist_ok=True)
    for name in ("ttgir", "llir", "amdgcn"):
        (folder / ("original." + name)).write_text(kernel.asm[name])
    (folder / "guarded.llir").write_text(guarded)
    (folder / "sites.json").write_text(json.dumps(sites, indent=2) + "\n")
    (folder / "bounds.json").write_text(
        json.dumps({NAMES[i]: b for i, b in bounds.items()}, indent=2) + "\n"
    )
    manager = get_override_manager(kernel.src.hash())
    Path(manager.cache_dir).mkdir(parents=True, exist_ok=True)
    destination = Path(manager.cache_dir, fn.__name__ + ".llir")
    destination.write_text(guarded)
    compiled = triton.compile(
        kernel.src, target=GPUTarget("hip", "gfx1250", 32), options=opts
    )
    assert compiled.asm["llir"] == guarded
    (folder / "guarded.amdgcn").write_text(compiled.asm["amdgcn"])
    destination.unlink()
    print(
        json.dumps(
            dict(
                prepared=side,
                sites=len(sites),
                lds_sites=sum(s["space"] == 3 for s in sites),
                shared=kernel.metadata.shared,
            )
        ),
        flush=True,
    )
    return compiled, records, dummy, sites, bounds


def summarize(side, prepared):
    _, records, _, sites, bounds = prepared
    data = records.cpu()
    bad = (data[:, :, 0] != 0).nonzero().tolist()
    summary = dict(side=side, bad_count=len(bad), first=[])
    for site, lane in bad[:32]:
        _, address, root, size = data[site, lane].tolist()
        base = bounds[root][0] if root >= 0 else 0
        summary["first"].append(
            dict(
                site=site,
                row=lane // 128,
                thread=lane % 128,
                tensor=NAMES[root] if root >= 0 else "LDS",
                address=hex(address),
                byte_offset=address - base,
                bytes=size,
                ir=sites[site]["source"],
            )
        )
    torch.save(data, ROOT / side / "records.pt")
    (ROOT / side / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    require_gpu0()
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT / "run.used").exists(), (
        "The one authorized isolation is already consumed"
    )
    knobs.cache.dir = str(ROOT / "cache")
    knobs.cache.override_dir = str(ROOT / "override")
    knobs.compilation.override = True
    knobs.compilation.always_compile = True
    d = checked_load("S6-inputs-7.pt")
    assert d["x"].shape == (8, 4096) and d["weight"].shape == (8192, 4096)
    # 512 tasks x 16 output columns; 8 K tiles x 512; 4 LDS ring slots.
    assert 512 * 16 == 8192 and 8 * 512 == 4096
    assert (4 - 1) * 16640 + 16624 == 66544
    (ROOT / "projection-bounds.json").write_text(
        json.dumps(
            dict(
                rows=8,
                N=8192,
                K=4096,
                tasks=512,
                tile_N=16,
                tile_K=512,
                lhs_descriptor_rows=16,
                intentional_zero_fill_rows=[8, 15],
                each_ring_bytes=66544,
                total_lds_bytes=133088,
                changed_since_passing_revision=False,
            ),
            indent=2,
        )
        + "\n"
    )
    fault = fault_module()
    tasks = []
    for side, mod in (("gluon", independent_tail), ("faulting_tlx", fault)):
        ws = mod.make_workspace("cuda")
        out = torch.empty_like(d["stream"], dtype=torch.bfloat16)
        updated = torch.empty_like(d["stream"])
        args = (
            ws.projection,
            d["stream"],
            d["post"],
            d["pre"],
            d["alpha"],
            d["beta"],
            out,
            updated,
            ws.state.status,
            None,
        )
        opts = dict(PRE_BF16=False, num_warps=4, enable_fp_fusion=False)
        if side == "faulting_tlx":
            opts["PIN_LAYOUT"] = True
        prepared = prepare(side, mod.finish, args, opts)
        tasks.append((side, ws, args, out, updated, prepared))
    torch.cuda.synchronize()
    # Exclusive marker before the first case kernel. Never retry after this.
    with (ROOT / "run.used").open("x") as file:
        file.write("seed7 normal sign+1; one guarded case\n")
    results = []
    for side, ws, args, out, updated, prepared in tasks:
        print("BEGIN " + side + " projection", flush=True)
        if side == "gluon":
            wide_projection.project[(512,)](
                d["x"],
                d["weight"],
                ws.projection,
                ws.state.status,
                None,
                N=8192,
                K=4096,
                STAGE=6,
                num_warps=2,
                enable_fp_fusion=False,
            )
        else:
            tlx_projection.project[(512,)](
                d["x"],
                d["weight"],
                ws.projection,
                ws.state.status,
                Failure=None,
                N=8192,
                K=4096,
                STAGE=6,
                num_warps=2,
                num_stages=1,
                enable_fp_fusion=False,
            )
        torch.cuda.synchronize()
        print("PASS " + side + " projection; BEGIN guarded boundary", flush=True)
        assert torch.all(ws.state.status == 1).item()
        constants = (False,) if side == "gluon" else (False, True)
        prepared[0][(8, 1, 1)](*args, *constants)
        torch.cuda.synchronize()
        result = summarize(side, prepared)
        results.append(result)
        torch.save(
            dict(out=out.cpu(), updated=updated.cpu(), projection=ws.projection.cpu()),
            ROOT / side / "outputs.pt",
        )
        if result["bad_count"]:
            (ROOT / "summary.json").write_text(
                json.dumps(dict(results=results, stopped_on_guard=True), indent=2)
                + "\n"
            )
            return
    comparisons = {}
    for name, index in (("out", 3), ("updated", 4)):
        a = tasks[0][index].cpu()
        b = tasks[1][index].cpu()
        dtype = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
        mismatch = (a.view(dtype) != b.view(dtype)).nonzero()
        comparisons[name] = dict(
            exact=len(mismatch) == 0, count=len(mismatch), first=mismatch[:1].tolist()
        )
    summary = dict(
        results=results,
        comparisons=comparisons,
        no_repeat=True,
        adopt_faulting_revision=False,
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

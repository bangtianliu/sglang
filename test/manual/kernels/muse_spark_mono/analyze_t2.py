"""CPU-only T2 exact-specialization inventories and three-level literal diffs."""

import difflib
import json
from pathlib import Path

import analyze_tlx_ir as ir

ROOT = Path("/home/bangtliu/coord/logs/phaseT-ir")
result = {}
for stage, phases in (
    (6, ("projection", "boundary")),
    (3, ("norm_gate", "projection", "AR_boundary")),
):
    for variant in ("before", "pinned"):
        ir.ROOT = ROOT / f"T2-chain-S{stage}-{variant}"
        result[f"S{stage}-{variant}"] = {}
        for phase in phases:
            pair = {side: ir.analyze(phase, side) for side in ("gluon", "tlx")}
            for data in pair.values():
                data["scalar_memory_ops"] = {
                    k: v
                    for k, v in data["opcodes"].items()
                    if k.startswith(("s_load_", "s_store_", "s_buffer_"))
                }
            for ext in ("ttgir", "llir", "amdgcn"):
                a, b = [
                    (ir.ROOT / phase / f"{s}.{ext}")
                    .read_text()
                    .splitlines(keepends=True)
                    for s in ("gluon", "tlx")
                ]
                (ir.ROOT / phase / f"gluon-vs-tlx.{ext}.diff").write_text(
                    "".join(difflib.unified_diff(a, b, fromfile="gluon", tofile="tlx"))
                )
            for side, d in pair.items():
                if phase == "projection":
                    assert sum(d["matrix_ops"].values()) == (128 if stage == 6 else 64)
                    assert d["memory_ops"]["tensor_load_to_lds"] == (
                        16 if stage == 6 else 8
                    )
                assert d["scoped_blocks_xcnt"]
            result[f"S{stage}-{variant}"][phase] = pair
        (ir.ROOT / "summary.json").write_text(
            json.dumps(result[f"S{stage}-{variant}"], indent=2) + "\n"
        )
(ROOT / "T2-inventory.json").write_text(json.dumps(result, indent=2) + "\n")
for label, phases in result.items():
    print(label)
    for phase, pair in phases.items():
        for side, d in pair.items():
            r = d["resources"]
            op = d["opcodes"]
            print(
                phase,
                side,
                json.dumps(
                    dict(
                        vgpr=r["vgprs"],
                        occ=r["occupancy"],
                        spills=r["spills"],
                        lds=r["lds"],
                        ds_load=op.get("ds_load_b128", 0),
                        ds_store=op.get("ds_store_b128", 0),
                        barriers=op.get("s_barrier_signal", 0),
                        memory=d["memory_ops"],
                        sync=d["sync_ops"],
                        policies=d["policies"],
                        convert=d["convert_ops"],
                    )
                ),
            )

"""CPU static S2 descriptor/fallback/merge diffs and cold chain interval."""

import difflib
import json
import random
import statistics
from pathlib import Path

import analyze_tlx_ir as ir

BASE = Path("/home/bangtliu/coord/logs")
root = BASE / "phaseT-ir"
folders = sorted(p.parent for p in (root / "T3-S2").glob("*/partial")) + [
    root / "T3-chain-S2-full"
]
for folder in folders:
    ir.ROOT = folder
    result = {}
    for phase in ("partial", "merge"):
        result[phase] = {side: ir.analyze(phase, side) for side in ("gluon", "tlx")}
        for ext in ("ttgir", "llir", "amdgcn"):
            a, b = [
                (folder / phase / f"{side}.{ext}").read_text().splitlines(keepends=True)
                for side in ("gluon", "tlx")
            ]
            (folder / phase / f"gluon-vs-tlx.{ext}.diff").write_text(
                "".join(difflib.unified_diff(a, b, fromfile="gluon", tofile="tlx"))
            )
        for side, d in result[phase].items():
            print(
                str(folder.relative_to(root)),
                phase,
                side,
                json.dumps(
                    {
                        k: d[k]
                        for k in (
                            "resources",
                            "memory_ops",
                            "matrix_ops",
                            "convert_ops",
                            "sync_ops",
                            "policies",
                        )
                    },
                    default=str,
                ),
            )
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
d = json.loads((BASE / "phaseT3-S2-S2-chain-131072--1.json").read_text())
rng = random.Random(20261118)
ci = {}
for order in ("AB", "BA"):
    blocks = []
    for i in range(0, len(d["raw"]), 4):
        rows = d["raw"][i : i + 4]
        if rows[0]["order"] == order:
            assert all(r["order"] == order for r in rows)
            blocks.append(
                {
                    s: [r["stage_us"] for r in rows if r["label"] == s]
                    for s in ("gluon", "tlx")
                }
            )
    ratios = []
    for _ in range(5000):
        sample = rng.choices(blocks, k=len(blocks))
        ratios.append(
            statistics.median(v for b in sample for v in b["tlx"])
            / statistics.median(v for b in sample for v in b["gluon"])
        )
    ratios.sort()
    ci[order] = dict(
        ratio=d["ratios"][order], blocks=len(blocks), ci95=[ratios[125], ratios[4874]]
    )
(BASE / "phaseT3-S2-chain-ci.json").write_text(json.dumps(ci, indent=2) + "\n")
print(json.dumps(ci))

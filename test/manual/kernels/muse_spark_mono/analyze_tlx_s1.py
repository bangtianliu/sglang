"""CPU-only SK8/IR audit and cold-chain confidence intervals for TLX S1."""

import argparse
import difflib
import json
import random
import re
import statistics
from pathlib import Path

import analyze_tlx_ir as ir

BASE = Path("/home/bangtliu/coord/logs")
ROOT = BASE / "phaseT-ir"
parser = argparse.ArgumentParser()
parser.add_argument("--label", default="")
a = parser.parse_args()
suffix = "-" + a.label if a.label else ""
for tag in ("T3-S1" + suffix, "T3-chain-S1" + suffix):
    ir.ROOT = ROOT / tag
    result = {}
    for phase in ("projection", "heads"):
        result[phase] = {side: ir.analyze(phase, side) for side in ("gluon", "tlx")}
        for ext in ("ttgir", "llir", "amdgcn"):
            a, b = [
                (ir.ROOT / phase / f"{side}.{ext}")
                .read_text()
                .splitlines(keepends=True)
                for side in ("gluon", "tlx")
            ]
            (ir.ROOT / phase / f"gluon-vs-tlx.{ext}.diff").write_text(
                "".join(difflib.unified_diff(a, b, fromfile="gluon", tofile="tlx"))
            )
        for side, d in result[phase].items():
            if phase == "projection":
                assert d["matrix_ops"] == {"v_wmma_f32_16x16x32_bf16": 256}
                assert d["memory_ops"]["tensor_load_to_lds"] == 32
                text = (ir.ROOT / phase / f"{side}.ttgir").read_text()
                dots = re.findall(r"^\s*(%\w+) = tt\.dot [^\n]*?, (%\w+) :", text, re.M)
                assert len(dots) == 16
                assert all(dots[i][1] == dots[i - 1][0] for i in range(1, 16, 2))
                assert len({dots[i][1] for i in range(0, 16, 2)}) == 1
                tree = {dots[2 * i + 1][0]: f"p{i}" for i in range(8)}
                for line in d["split_tree_lines"]:
                    out, a, b = re.search(
                        r"(%\w+) = arith.addf (%\w+), (%\w+)", line
                    ).groups()
                    tree[out] = "(" + tree[a] + "+" + tree[b] + ")"
                assert tree[out] == "(((p0+p4)+(p2+p6))+((p1+p5)+(p3+p7)))"
                d["fixed_tree"] = tree[out]
            print(
                tag,
                phase,
                side,
                json.dumps(
                    {
                        k: d[k]
                        for k in (
                            "resources",
                            "memory_ops",
                            "sync_ops",
                            "matrix_ops",
                            "convert_ops",
                            "policies",
                        )
                    },
                    default=str,
                ),
            )
    (ir.ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
d = json.loads(
    (BASE / ("phaseT3-S1" + suffix + "-S1-chain-8192-2047.json")).read_text()
)
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
        sampled = rng.choices(blocks, k=len(blocks))
        ratios.append(
            statistics.median(v for b in sampled for v in b["tlx"])
            / statistics.median(v for b in sampled for v in b["gluon"])
        )
    ratios.sort()
    ci[order] = dict(
        ratio=d["ratios"][order], blocks=len(blocks), ci95=[ratios[125], ratios[4874]]
    )
(BASE / ("phaseT3-S1" + suffix + "-chain-ci.json")).write_text(
    json.dumps(ci, indent=2) + "\n"
)
print(json.dumps(ci))

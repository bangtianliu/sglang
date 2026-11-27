"""CPU-only T1 S6 three-level IR diff and paired block confidence intervals."""

import difflib
import json
import random
import re
import statistics
from pathlib import Path

import analyze_tlx_ir as ir

ROOT = Path("/home/bangtliu/coord/logs/phaseT-ir/T1-S6")


def main():
    ir.ROOT = ROOT
    result = {}
    for phase in ("projection", "boundary"):
        result[phase] = {s: ir.analyze(phase, s) for s in ("gluon", "tlx")}
        for ext in ("ttgir", "llir", "amdgcn"):
            a, b = [
                (ROOT / phase / f"{s}.{ext}").read_text().splitlines(keepends=True)
                for s in ("gluon", "tlx")
            ]
            (ROOT / phase / f"gluon-vs-tlx.{ext}.diff").write_text(
                "".join(difflib.unified_diff(a, b, fromfile="gluon", tofile="tlx"))
            )
    for side in ("gluon", "tlx"):
        projection = result["projection"][side]
        assert projection["matrix_ops"] == {"v_wmma_f32_16x16x32_bf16": 128}
        assert projection["memory_ops"]["tensor_load_to_lds"] == 16
        text = (ROOT / "projection" / f"{side}.ttgir").read_text()
        dots = re.findall(r"^\s*(%\w+) = tt\.dot [^\n]*?, (%\w+) :", text, re.M)
        assert len(dots) == 8
        assert all(dots[i][1] == dots[i - 1][0] for i in (1, 3, 5, 7))
        assert len({dots[i][1] for i in (0, 2, 4, 6)}) == 1
        tree = {dots[2 * i + 1][0]: f"p{i}" for i in range(4)}
        for line in projection["split_tree_lines"]:
            out, left, right = re.search(
                r"(%\w+) = arith.addf (%\w+), (%\w+)", line
            ).groups()
            tree[out] = "(" + tree[left] + "+" + tree[right] + ")"
        projection["fixed_tree"] = tree[out]
        assert projection["fixed_tree"] == "((p0+p2)+(p1+p3))"
        boundary = result["boundary"][side]
        assert boundary["scoped_asm_blocks"] == 2
        assert boundary["policies"]["scope_DEV"] == 16
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    p = ROOT.parent.parent / "phaseT1-S6-chain-8192--1.json"
    d = json.loads(p.read_text())
    rng = random.Random(20261118)
    ci = {}
    for order in ("AB", "BA"):
        blocks = []
        for i in range(0, len(d["raw"]), 4):
            rows = d["raw"][i : i + 4]
            if rows[0]["order"] == order:
                assert len(rows) == 4 and all(r["order"] == order for r in rows)
                blocks.append(
                    {
                        s: [r["stage_us"] for r in rows if r["label"] == s]
                        for s in ("gluon", "tlx")
                    }
                )
        ratios = []
        for _ in range(5000):
            sampled = rng.choices(blocks, k=len(blocks))
            values = {s: [v for b in sampled for v in b[s]] for s in ("gluon", "tlx")}
            ratios.append(
                statistics.median(values["tlx"]) / statistics.median(values["gluon"])
            )
        ratios.sort()
        ci[order] = dict(
            blocks=len(blocks),
            ratio=d["ratios"][order],
            ci95=[ratios[125], ratios[4874]],
        )
    (ROOT / "chain-ci.json").write_text(json.dumps(ci, indent=2) + "\n")
    print(json.dumps(ci))
    for phase, sides in result.items():
        print(phase)
        for side, x in sides.items():
            print(
                side,
                json.dumps(
                    {
                        k: x[k]
                        for k in (
                            "resources",
                            "memory_ops",
                            "sync_ops",
                            "matrix_ops",
                            "convert_ops",
                            "policies",
                            "scoped_asm_blocks",
                        )
                    },
                    default=str,
                ),
            )


if __name__ == "__main__":
    main()

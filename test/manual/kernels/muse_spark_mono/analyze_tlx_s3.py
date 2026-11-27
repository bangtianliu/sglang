"""CPU-only literal T1 S3 IR differences and non-vacuous scope/XCNT audit."""

import difflib
import json
import re
from pathlib import Path

import analyze_tlx_ir as ir

ROOT = Path("/home/bangtliu/coord/logs/phaseT-ir/T1-S3")


def main():
    ir.ROOT = ROOT
    result = {}
    for phase in ("norm_gate", "projection", "AR_boundary"):
        result[phase] = {side: ir.analyze(phase, side) for side in ("gluon", "tlx")}
        for ext in ("ttgir", "llir", "amdgcn"):
            a, b = [
                (ROOT / phase / f"{s}.{ext}").read_text().splitlines(keepends=True)
                for s in ("gluon", "tlx")
            ]
            (ROOT / phase / f"gluon-vs-tlx.{ext}.diff").write_text(
                "".join(difflib.unified_diff(a, b, fromfile="gluon", tofile="tlx"))
            )
    for side in ("gluon", "tlx"):
        p = result["projection"][side]
        assert p["matrix_ops"] == {"v_wmma_f32_16x16x32_bf16": 64}
        assert p["memory_ops"]["tensor_load_to_lds"] == 8
        text = (ROOT / "projection" / f"{side}.ttgir").read_text()
        dots = re.findall(r"^\s*(%\w+) = tt\.dot [^\n]*?, (%\w+) :", text, re.M)
        assert len(dots) == 4 and len({d[1] for d in dots}) == 1
        tree = {dots[i][0]: f"p{i}" for i in range(4)}
        for line in p["split_tree_lines"]:
            out, a, b = re.search(r"(%\w+) = arith.addf (%\w+), (%\w+)", line).groups()
            tree[out] = "(" + tree[a] + "+" + tree[b] + ")"
        assert tree[out] == "((p0+p2)+(p1+p3))"
        p["fixed_tree"] = tree[out]
        a = result["AR_boundary"][side]
        assert a["scoped_asm_blocks"] > 0 and a["scoped_blocks_xcnt"]
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for phase, sides in result.items():
        print(phase)
        for side, d in sides.items():
            print(
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
                            "scoped_asm_blocks",
                        )
                    },
                    default=str,
                ),
            )


if __name__ == "__main__":
    main()

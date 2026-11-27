"""Static scratch instruction attribution through assembler inline source chains."""

import ast
import collections
import json
import re
from pathlib import Path

LOG = Path("/home/bangtliu/coord/logs")


def stage_ranges(source):
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_launch"
    )
    ranges = []
    for n in fn.body:
        if isinstance(n, ast.If) and "ONLY_STAGE" in ast.unparse(n.test):
            # Five barrier wrappers only test ONLY_STAGE ==0 and are labeled separately.
            match = re.search(r"ONLY_STAGE == ([1-6])", ast.unparse(n.test))
            if match:
                ranges.append((n.lineno, n.end_lineno, "S" + match[1]))
            else:
                ranges.append((n.lineno, n.end_lineno, "grid-barrier"))
    return ranges


def analyze(path, ranges, rootname):
    files = {}
    location = ""
    last_stage = "setup/unattributed"
    block = "entry"
    records = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        file = re.match(r'\s*\.file\s+(\d+)\s+"([^"]+)"\s+"([^"]+)"', line)
        if file:
            files[file[1]] = file[2] + "/" + file[3]
        if re.match(r"^\.LBB\d+_\d+:", line):
            block = line.split(":")[0]
        if line.lstrip().startswith(".loc"):
            location = line.strip()
            roots = re.findall(re.escape(rootname) + r":(\d+)", line)
            root = int(roots[-1]) if roots else 0
            if root:
                last_stage = next(
                    (s for first, last, s in ranges if first <= root <= last),
                    "setup/unattributed",
                )
        hit = re.search(r"\b(scratch_(load|store)_b(8|16|32|64|96|128))\b", line)
        if hit:
            parsed = re.match(r"\.loc\s+(\d+)\s+(\d+)\s+", location)
            file = files.get(parsed[1], "unknown") if parsed else "unknown"
            source_line = int(parsed[2]) if parsed else 0
            records.append(
                dict(
                    asm_line=i + 1,
                    opcode=hit[1],
                    operation=hit[2],
                    width_bytes=int(hit[3]) // 8,
                    stage_context=last_stage,
                    source=file,
                    source_line=source_line,
                    loc=location,
                    block=block,
                    instruction=line.strip(),
                    attribution="inline root source location; last nonzero root retained across compiler line0",
                )
            )
    counts = collections.defaultdict(collections.Counter)
    sites = collections.Counter()
    for r in records:
        counts[r["stage_context"]][r["opcode"]] += 1
        sites[(r["stage_context"], r["source"], r["source_line"])] += 1
    return dict(
        static_scratch_instructions=len(records),
        by_stage={k: dict(v) for k, v in counts.items()},
        top_sites=[
            dict(stage=s, file=f, line=l, count=n)
            for (s, f, l), n in sites.most_common()
        ],
        records=records,
        private_segment_lines=[
            x.strip() for x in lines if "private_segment_fixed_size" in x
        ],
        interpretation="Static instruction sites, not dynamic memory traffic or a runtime profile. Stage context from complete inline debug chain; line0 spill placement uses neighboring nonzero root context.",
    )


def main():
    source = (LOG / "phase46-layer46-source.py").read_text()
    ranges = stage_ranges(source)
    result = {}
    for label in ["masked_all", *[f"S{s}_standalone" for s in range(1, 7)]]:
        result[label] = analyze(LOG / f"phase46-{label}.amdgcn", ranges, "layer46.py")
    # The untouched default has ordinary per-stage stamp boundaries.
    source = (LOG / "phase46-accepted-source.py").read_text()
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_launch"
    )
    starts = []
    for n in fn.body:
        if (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "stamp"
            and isinstance(n.value.args[-1], ast.Constant)
            and n.value.args[-1].value in (0, 2, 4, 6, 8, 10)
        ):
            starts.append(n.lineno)
    ranges = [
        (first, (starts[i + 1] - 1 if i + 1 < 6 else fn.end_lineno), f"S{i + 1}")
        for i, first in enumerate(starts)
    ]
    result["accepted"] = analyze(
        LOG / "phase46-accepted.amdgcn", ranges, "layer_accepted.py"
    )
    (LOG / "phase46-spills.json").write_text(json.dumps(result, indent=2) + "\n")
    for label, d in result.items():
        print(label, d["static_scratch_instructions"], d["by_stage"])
        print("top sites", d["top_sites"][:5])


if __name__ == "__main__":
    main()

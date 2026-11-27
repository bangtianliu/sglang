"""Application-boundary import audit for the all-TLX production entry.

Stock Triton/uTLX compatibility imports are vendor internals, permitted by T5.
Checks parent initializers and every loaded SGLang module, and a transitive
static module graph over the application implementation, including lazy imports.
"""

import ast
import hashlib
import importlib
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "python"
ENTRY = "sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_chain"
PREFIX = "sglang.kernels.plugin_kernels.amd.muse_spark"
OUT = Path("/home/bangtliu/coord/logs/phaseT5-import-audit.json")


def path(name):
    p = ROOT.joinpath(*name.split("."))
    for candidate in (p.with_suffix(".py"), p / "__init__.py"):
        if candidate.is_file():
            return candidate


def targets(module, file, node):
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    base = node.module or ""
    if node.level:
        package = module if file.name == "__init__.py" else module.rsplit(".", 1)[0]
        base = importlib.util.resolve_name("." * node.level + base, package)
    return [base] + [base + "." + a.name for a in node.names if a.name != "*"]


def main():
    entry = importlib.import_module(ENTRY)
    assert entry.ALL_TLX == (1, 2, 3, 4, 5, 6)
    loaded = {
        n: m.__file__
        for n, m in sys.modules.items()
        if n.startswith("sglang")
        and getattr(m, "__file__", None)
        and str(m.__file__).startswith(str(ROOT))
    }
    queue = deque([ENTRY])
    seen = set()
    edges = []
    errors = []
    while queue:
        module = queue.popleft()
        if module in seen:
            continue
        file = path(module)
        if file is None:
            continue
        seen.add(module)
        for node in ast.walk(ast.parse(file.read_text())):
            for target in targets(module, file, node):
                if target.startswith("triton.experimental.gluon"):
                    errors.append(dict(source=module, line=node.lineno, target=target))
                if target.startswith(PREFIX) and path(target):
                    edges.append(dict(source=module, line=node.lineno, target=target))
                    queue.append(target)
    # Parent initializers and other actual SGLang imports are checked fully too.
    for module, file in loaded.items():
        for node in ast.walk(ast.parse(Path(file).read_text())):
            for target in targets(module, Path(file), node):
                if target.startswith("triton.experimental.gluon"):
                    errors.append(dict(source=module, line=node.lineno, target=target))
    forbidden = {
        "chained",
        "common_qkvg",
        "common_attention",
        "common_attn_tail",
        "common_route",
        "common_experts",
        "common_tail",
        "independent_qkvg",
        "independent_attn_tail",
        "independent_experts",
        "independent_tail",
        "vector_memory",
        "standin",
        "collective_fast",
        "packed_memory",
        "scoped",
        "sync",
        "timeline",
        "streamed_gemm",
    }
    assert not [n for n in seen if n.rsplit(".", 1)[-1] in forbidden], seen
    assert not errors, errors
    result = dict(
        entry=ENTRY,
        all_stages=entry.ALL_TLX,
        application_modules=[
            dict(
                module=n,
                path=str(path(n)),
                sha256=hashlib.sha256(path(n).read_bytes()).hexdigest(),
            )
            for n in sorted(seen)
        ],
        loaded_own_modules=loaded,
        edges=edges,
        violations=errors,
        vendor_boundary="Stock Triton/uTLX compiler compatibility internals allowed; no application Gluon modules or launched Gluon kernels",
        passed=True,
    )
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=True,
                application_modules=len(seen),
                loaded_own_modules=len(loaded),
            )
        )
    )


if __name__ == "__main__":
    main()

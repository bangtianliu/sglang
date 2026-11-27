"""CPU-only Gluon symbol/signature inventory for both validated toolchains."""

import ast
import importlib
import inspect
import json
from pathlib import Path

ROOT = Path("/home/bangtliu/sglang-mono")


def inventory():
    symbols = {}
    roots = (
        ROOT / "python/sglang/kernels/plugin_kernels/amd/muse_spark",
        ROOT / "test/manual/kernels/muse_spark_mono",
    )
    for base in roots:
        for path in base.rglob("*.py"):
            tree = ast.parse(path.read_text())
            aliases = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        full = node.module + "." + alias.name
                        if full.startswith("triton.experimental.gluon"):
                            aliases[alias.asname or alias.name] = full
                            symbols.setdefault(full, []).append(
                                f"{path.relative_to(ROOT)}:{node.lineno}"
                            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                attrs = []
                head = node
                while isinstance(head, ast.Attribute):
                    attrs.append(head.attr)
                    head = head.value
                if isinstance(head, ast.Name) and head.id in aliases:
                    full = aliases[head.id] + "." + ".".join(reversed(attrs))
                    symbols.setdefault(full, []).append(
                        f"{path.relative_to(ROOT)}:{node.lineno}"
                    )
    return symbols


def resolve(full):
    parts = full.split(".")
    for end in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:end]))
        except ModuleNotFoundError:
            continue
        for attr in parts[end:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(full)


def main():
    import triton

    rows = []
    for name, uses in sorted(inventory().items()):
        try:
            value = resolve(name)
            try:
                signature = str(inspect.signature(value))
            except (TypeError, ValueError):
                signature = None
            row = dict(
                symbol=name, present=True, signature=signature, uses=sorted(set(uses))
            )
        except (AttributeError, ImportError) as e:
            row = dict(symbol=name, present=False, error=str(e), uses=sorted(set(uses)))
        rows.append(row)
    print(json.dumps(dict(triton=triton.__version__, symbols=rows), indent=2))
    if any(not r["present"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

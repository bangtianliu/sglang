"""Keep this image's ROCm SDK LLVM symbols local before plugin-enabled Triton.

Torch's ROCm SDK bootstrap defaults to global preloads, which collide with
Triton's exported LLVM symbols. Populate the SDK's handle cache using its
public local-preload API before Torch initializes. This file is installed only
in the derived benchmark image; it does not modify Torch or Triton sources.
"""

import ast
from importlib.util import find_spec
from pathlib import Path

import rocm_sdk

_torch_root = Path(find_spec("torch").origin).parent
_bootstrap = ast.parse((_torch_root / "_rocm_init.py").read_text())
_preloads = next(
    ast.literal_eval(keyword.value)
    for node in ast.walk(_bootstrap)
    if isinstance(node, ast.Call)
    for keyword in node.keywords
    if keyword.arg == "preload_shortnames"
)
rocm_sdk.preload_libraries(*_preloads, rtld_global=False)

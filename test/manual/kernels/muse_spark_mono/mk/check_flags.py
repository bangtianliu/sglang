"""Run check_tlx_chain.py with stage-module flags set: FLAGS="tlx_experts.FUSE13_2=1,..."
(values are Python literals). Other args pass through to the suite."""
import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import check_tlx_chain  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250 import stages  # noqa: E402
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_experts  # noqa: E402,F401

for kv in filter(None, os.environ.get("FLAGS", "").split(",")):
    k, v = kv.split("=")
    mod, attr = k.split(".")
    setattr(getattr(stages, mod), attr, ast.literal_eval(v))
    print("flag", mod, attr, getattr(getattr(stages, mod), attr), flush=True)
check_tlx_chain.main()

"""Load the preserved faulting boundary without changing the adopted modules."""

import importlib.util
from pathlib import Path

ROOT = Path(
    "/home/bangtliu/coord/logs/phaseT2-fault-source/python/sglang/kernels/plugin_kernels/amd/muse_spark/gfx1250/stages"
)


def fault_module():
    def load(name):
        spec = importlib.util.spec_from_file_location(
            "t2_fault_" + name, ROOT / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    boundary = load("tlx_boundary")
    tail = load("tlx_tail")
    tail.norm_residual = boundary.norm_residual
    return tail

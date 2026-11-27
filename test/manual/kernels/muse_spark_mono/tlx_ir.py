"""Save exposed IR levels and exact-kernel resources for T0 feasibility proofs."""

import json
from pathlib import Path

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)

ROOT = Path("/home/bangtliu/coord/logs/phaseT-ir")


def dump(kernel, stage, side):
    dest = ROOT / stage
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("ttgir", "llir", "amdgcn"):
        (dest / f"{side}.{name}").write_text(kernel.asm[name])
    if "ttir" in kernel.asm:
        (dest / f"{side}.ttir").write_text(kernel.asm["ttir"])
    r = kernel_occupancy(stage + "-" + side, kernel)
    result = dict(
        vgprs=kernel.n_regs,
        spills=kernel.n_spills,
        lds=kernel.metadata.shared,
        waves=kernel.metadata.num_warps,
        occupancy=r.active_ctas_per_cu,
        ir_keys=list(kernel.asm),
        metadata=kernel.metadata._asdict(),
    )
    result = json.loads(json.dumps(result, default=str))
    (dest / f"{side}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

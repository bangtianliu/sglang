"""GPU0-only dissection of the unchanged native Triton3.9 A16W4 path."""

import json
from pathlib import Path

import torch
import triton
from bench_hbm import graph_times
from check_experts import cases, load
from layer_fixture import require_gpu0

from sglang.srt.rocm_muse_spark import muse_moe_triton as native

ROOT = Path("/home/bangtliu/coord/logs")


class Capture:
    def __init__(self, kernel):
        self.kernel = kernel
        self.records = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            result = self.kernel[grid](*args, **kwargs)
            self.records.append((grid, kwargs, result))
            return result

        return run


def main():
    require_gpu0()
    assert triton.__version__.split(".")[:2] == ["3", "9"]
    x, w = load(7)
    ids = cases(7, x.device)["typical"]
    table = native.moe_align_block_size_small(ids, 64, 0, 16)
    inter = torch.full((64, 4096), float("nan"), device=x.device, dtype=torch.bfloat16)
    out = torch.full_like(inter, float("nan"))
    cfg = native._pick_config(8, 8, 64, True)
    capture = Capture(native.fused_moe_gemm_kernel)
    native.fused_moe_gemm_kernel = capture
    records = []
    for name, a, b, s, c, first in [
        ("w13", x, w["w13"], w["w13_scale"], inter, True),
        ("w2", inter, w["w2"], w["w2_scale"], out, False),
    ]:

        def call():
            native._launch_gemm(
                a, b, s, c, *table, 4096, 4096, 64, 8, first, first, True, cfg
            )

        call()
        torch.cuda.synchronize()
        grid, kwargs, k = capture.records[-1]
        for fmt in ("amdgcn", "ttgir", "llir"):
            (ROOT / f"phase3e1e-native-{name}.{fmt}").write_text(k.asm[fmt])
        record = dict(
            phase=name,
            grid=grid,
            active_ctas=int(table[2].item() // 16 * 64),
            config=cfg,
            launch_kwargs=kwargs,
            registers=k.n_regs,
            spills=k.n_spills,
            shared=k.metadata.shared,
            metadata=k.metadata._asdict(),
            time=graph_times(call, samples=30),
        )
        print(json.dumps(record, default=str), flush=True)
        records.append(record)
        capture.records.clear()
    valid = ids.flatten() < 64
    assert inter[valid].isfinite().all() and out[valid].isfinite().all()
    ref = torch.load(ROOT / "phase3e1-native-7.pt", weights_only=True)["typical"]
    assert torch.equal(inter[valid], ref["inter"].cuda()[valid])
    assert torch.equal(out[valid], ref["y"].cuda()[valid])
    (ROOT / "phase3e1e-native.json").write_text(
        json.dumps(records, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()

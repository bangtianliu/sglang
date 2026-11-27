"""Bounded native S5 health check against partition-independent FP64."""

import json
from pathlib import Path

import torch
import triton
from layer_fixture import build_layer, require_gpu0

from sglang.srt.rocm_muse_spark import muse_moe_triton as native


def error(a, b):
    d = (a.double() - b.double()).abs()
    return dict(
        max_abs=d.max().item(),
        relative_l2=(d.square().sum() / b.double().square().sum()).sqrt().item(),
    )


def main():
    require_gpu0()
    if triton.__version__.split(".")[:2] != ["3", "9"]:
        raise RuntimeError(
            "native expert health/reference requires Triton 3.9; 3.8 faulted in the approved probe"
        )
    f = build_layer(seed=7, components=("experts",))
    x = (f.x[:, :4096] * 0.5).contiguous()
    w = f.weights
    ids = torch.full((8, 8), 255, device="cuda", dtype=torch.int32)
    ids[:, 0] = 3
    sorted_ids, expert_ids, num_post = native.moe_align_block_size_small(ids, 64, 0, 16)
    print(
        "align",
        sorted_ids[:16].tolist(),
        expert_ids[:4].tolist(),
        num_post.item(),
        flush=True,
    )
    inter = torch.full((64, 4096), float("nan"), device="cuda", dtype=torch.bfloat16)
    y = torch.full_like(inter, float("nan"))
    cfg = native._pick_config(8, 8, 64, True)
    native._launch_gemm(
        x,
        w["w13"],
        w["w13_scale"],
        inter,
        sorted_ids,
        expert_ids,
        num_post,
        4096,
        4096,
        64,
        8,
        True,
        True,
        True,
        cfg,
    )
    torch.cuda.synchronize()
    print("W13 complete", flush=True)
    native._launch_gemm(
        inter,
        w["w2"],
        w["w2_scale"],
        y,
        sorted_ids,
        expert_ids,
        num_post,
        4096,
        4096,
        64,
        8,
        False,
        False,
        True,
        cfg,
    )
    torch.cuda.synchronize()
    print("W2 complete", flush=True)
    dense = native.dequant_mxfp4(w["w13"][3], w["w13_scale"][3]).double()
    gu = x.double() @ dense.T
    gate, up = gu.split(4096, dim=-1)
    ref = (gate.sigmoid() * gate * up).bfloat16()
    dense2 = native.dequant_mxfp4(w["w2"][3], w["w2_scale"][3]).double()
    ref2 = (inter[::8].double() @ dense2.T).bfloat16()
    full = (ref.double() @ dense2.T).bfloat16()
    row = dict(
        torch=str(torch.__version__),
        triton=__import__("triton").__version__,
        fp4_mode=native.FP4_MODE,
        inter=error(inter[::8], ref),
        w2_matched=error(y[::8], ref2),
        w2_end_to_end=error(y[::8], full),
        finite=bool(torch.isfinite(y[::8]).all()),
    )
    print(json.dumps(row), flush=True)
    Path("/home/bangtliu/coord/logs/phase3e1-native-health.json").write_text(
        json.dumps(row, indent=2) + "\n"
    )
    assert (
        row["finite"]
        and row["inter"]["relative_l2"] < 0.01
        and row["w2_matched"]["relative_l2"] < 0.01
    )


if __name__ == "__main__":
    main()

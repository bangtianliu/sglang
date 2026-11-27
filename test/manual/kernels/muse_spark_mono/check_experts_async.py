"""GPU0 standalone four-wave async expert port: saved native/FP64 and resources."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from bench_hbm import graph_times
from check_expert_native import error
from check_experts import cases
from check_experts_routed import oracle, single_eight
from check_experts_valu import mixed
from layer_fixture import require_gpu0
from reference_archive import checked_load, gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import experts_async
from sglang.srt.rocm_muse_spark import muse_moe_triton as native

ROOT = Path("/home/bangtliu/coord/logs")


def run(seed, selected=None):
    saved = checked_load(f"phase3e1-inputs-{seed}.pt")
    x, w = saved["x"], saved["weights"]
    refs = checked_load(f"phase3e1-native-{seed}.pt", device=False)
    refs["one8"] = checked_load(f"phase3e1b-native-one8-{seed}.pt", device=False)
    refs["mixed"] = checked_load(f"phase3e1c-native-mixed-{seed}.pt", device=False)
    patterns = cases(seed, "cuda")
    patterns["one8"] = single_eight("cuda")
    patterns["mixed"] = mixed("cuda")
    inter = torch.empty((64, 4096), device=x.device, dtype=torch.bfloat16)
    output = torch.empty_like(inter)
    result_path = ROOT / f"phase3e1f2-results-{seed}.json"
    records = (
        json.loads(result_path.read_text()) if selected and result_path.exists() else []
    )
    for name, ids in patterns.items():
        if selected and name != selected:
            continue
        table = native.moe_align_block_size_small(ids, 64, 0, 16)
        inter.fill_(float("nan"))
        output.fill_(float("nan"))

        def call():
            return experts_async.experts(
                x,
                *table,
                w["w13"],
                w["w13_scale"],
                w["w2"],
                w["w2_scale"],
                inter,
                output,
            )

        kernels = call()
        torch.cuda.synchronize()
        resources = []
        for phase, k in zip(("w13", "w2"), kernels):
            resource = dict(
                phase=phase,
                registers=k.n_regs,
                spills=k.n_spills,
                shared=k.metadata.shared,
                occupancy=kernel_occupancy(phase, k).active_ctas_per_cu,
            )
            print(
                json.dumps(dict(event="resources", seed=seed, case=name, **resource)),
                flush=True,
            )
            resources.append(resource)
            for fmt in ("amdgcn", "ttgir"):
                (ROOT / f"phase3e1f2-{phase}.{fmt}").write_text(k.asm[fmt])
        valid = (ids.flatten() >= 0) & (ids.flatten() < 64)
        assert inter[~valid].isnan().all() and output[~valid].isnan().all()
        comparison = None
        fp64 = []
        if valid.any():
            ref = gpu(refs[name])
            comparison = dict(
                inter=error(inter[valid], ref["inter"][valid]),
                output=error(output[valid], ref["y"][valid]),
            )
            print(
                json.dumps(
                    dict(
                        event="native-comparison",
                        seed=seed,
                        case=name,
                        errors=comparison,
                    )
                ),
                flush=True,
            )
            assert comparison["inter"]["relative_l2"] < 0.01
            assert comparison["output"]["relative_l2"] < 0.015
            fp64 = oracle(x, ids, w, SimpleNamespace(inter=inter, output=output))
        timing = graph_times(call)
        record = dict(
            seed=seed,
            case=name,
            resources=resources,
            errors=comparison,
            oracle=fp64,
            time=timing,
            native39=refs[name]["time"],
        )
        print(json.dumps(record), flush=True)
        records.append(record)
        (ROOT / f"phase3e1f2-results-{seed}.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
    return records


def main():
    require_gpu0()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--case", choices=("typical", "one", "distinct", "empty", "one8", "mixed")
    )
    args = parser.parse_args()
    run(args.seed, args.case)


if __name__ == "__main__":
    main()

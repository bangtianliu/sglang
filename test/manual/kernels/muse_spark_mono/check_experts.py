"""S5 one-rank validation; native references use saved identical tensors in 3.9."""

import argparse
import json
from pathlib import Path

import torch
import triton
from bench_hbm import graph_times, stream_read
from check_expert_native import error
from layer_fixture import build_layer, require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import experts as stage
from sglang.srt.rocm_muse_spark import muse_moe_triton as native

ROOT = Path("/home/bangtliu/coord/logs")
SEEDS = (7, 20261117)


def cases(seed, device):
    gen = torch.Generator().manual_seed(seed)
    typical = torch.randperm(256, generator=gen)[:64].reshape(8, 8).to(torch.int32)
    return {
        name: ids.to(device)
        for name, ids in {
            "typical": typical,
            "one": torch.full((8, 8), 3, dtype=torch.int32),
            "distinct": torch.arange(64, dtype=torch.int32).reshape(8, 8),
            "empty": torch.full((8, 8), 255, dtype=torch.int32),
        }.items()
    }


def prepare():
    for seed in SEEDS:
        f = build_layer(seed=seed, components=("experts",))
        torch.save(
            dict(
                x=(f.x[:, :4096] * 0.5).contiguous().cpu(),
                weights={k: v.cpu() for k, v in f.weights.items()},
            ),
            ROOT / f"phase3e1-inputs-{seed}.pt",
        )
        print("prepared", seed, flush=True)


def load(seed):
    f = torch.load(ROOT / f"phase3e1-inputs-{seed}.pt", weights_only=True)
    return f["x"].cuda(), {k: v.cuda() for k, v in f["weights"].items()}


def native_reference():
    if triton.__version__.split(".")[:2] != ["3", "9"]:
        raise RuntimeError(
            "native expert health/reference requires Triton 3.9; 3.8 faulted in the approved probe"
        )
    for seed in SEEDS:
        x, w = load(seed)
        records = {}
        for name, ids in cases(seed, "cuda").items():
            sorted_ids, expert_ids, num_post = native.moe_align_block_size_small(
                ids, 64, 0, 16
            )
            inter = torch.full(
                (64, 4096), float("nan"), device="cuda", dtype=torch.bfloat16
            )
            y = torch.full_like(inter, float("nan"))
            cfg = native._pick_config(8, 8, 64, True)

            def call():
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

            call()
            torch.cuda.synchronize()
            records[name] = dict(
                inter=inter.cpu(),
                y=y.cpu(),
                time=graph_times(call),
                torch=str(torch.__version__),
                triton=__import__("triton").__version__,
            )
            print(
                json.dumps(dict(seed=seed, case=name, time=records[name]["time"])),
                flush=True,
            )
        torch.save(records, ROOT / f"phase3e1-native-{seed}.pt")


def validate():
    records = []
    for seed in SEEDS:
        x, w = load(seed)
        reference = torch.load(ROOT / f"phase3e1-native-{seed}.pt", weights_only=True)
        ws = stage.make_workspace(x.device)
        for name, ids in cases(seed, "cuda").items():
            ws.inter.fill_(float("nan"))
            ws.output.fill_(float("nan"))
            call = lambda: stage.experts(
                x, ids, w["w13"], w["w13_scale"], w["w2"], w["w2_scale"], ws
            )
            compiled = call()
            torch.cuda.synchronize()
            assert ws.status.eq(1).all()
            present = ids[ids < 64].unique().tolist()
            oracle1 = []
            oracle2 = []
            end = []
            for e in present:
                dense = native.dequant_mxfp4(w["w13"][e], w["w13_scale"][e]).double()
                gate, up = (x.double() @ dense.T).split(4096, -1)
                ref = (gate.sigmoid() * gate * up).bfloat16()
                dense2 = native.dequant_mxfp4(w["w2"][e], w["w2_scale"][e]).double()
                matched = (ws.inter[e].double() @ dense2.T).bfloat16()
                full = (ref.double() @ dense2.T).bfloat16()
                oracle1.append(error(ws.inter[e], ref))
                oracle2.append(error(ws.output[e], matched))
                end.append(error(ws.output[e], full))
                assert (
                    oracle1[-1]["relative_l2"] < 0.01
                    and oracle2[-1]["relative_l2"] < 0.01
                )
            valid = ids.flatten() < 64
            token = (torch.arange(64, device="cuda") // 8)[valid]
            expert = ids.flatten()[valid].long()
            nr = reference[name]
            n1 = (
                error(ws.inter[expert, token], nr["inter"].cuda()[valid])
                if present
                else None
            )
            n2 = (
                error(ws.output[expert, token], nr["y"].cuda()[valid])
                if present
                else None
            )
            if present:
                assert n1["relative_l2"] < 0.01 and n2["relative_l2"] < 0.015
            else:
                assert ws.inter.isnan().all() and ws.output.isnan().all()
            timing = graph_times(call)
            assert ws.status.eq(1).all()
            if name == "typical":
                (ROOT / f"phase3e1-final-stage-{seed}.amdgcn").write_text(
                    compiled.asm["amdgcn"]
                )
            read_bytes = len(present) * 26738688
            if read_bytes:
                data = torch.zeros(read_bytes, device="cuda", dtype=torch.uint8)
                sink = torch.empty(
                    (read_bytes + 4095) // 4096, device="cuda", dtype=torch.int32
                )
                read = graph_times(
                    lambda: stream_read[(sink.numel(),)](data, sink, read_bytes, 4096)
                )
                del data, sink
            else:
                read = None
            record = dict(
                seed=seed,
                case=name,
                distinct=len(present),
                read_bytes=read_bytes,
                stage=timing,
                streaming_read=read,
                native39=nr["time"],
                oracle_inter=oracle1,
                oracle_w2_matched=oracle2,
                oracle_end=end,
                native_inter=n1,
                native_output=n2,
                registers=compiled.n_regs,
                spills=compiled.n_spills,
                shared=compiled.metadata.shared,
            )
            print(json.dumps(record), flush=True)
            records.append(record)
    (ROOT / "phase3e1-results.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--native-reference", action="store_true")
    args = p.parse_args()
    if args.prepare:
        prepare()
    elif args.native_reference:
        native_reference()
    else:
        validate()


if __name__ == "__main__":
    main()

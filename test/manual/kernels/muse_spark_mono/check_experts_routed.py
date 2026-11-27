"""3e.1b routed-M16 W13/W2 correctness, zero-spill checks and latency."""

import argparse
import json
from pathlib import Path

import torch
import triton
from bench_hbm import graph_times, stream_read
from check_expert_native import error
from check_experts import SEEDS, cases, load
from layer_fixture import require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    experts as baseline,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    experts_routed as stage,
)
from sglang.srt.rocm_muse_spark import muse_moe_triton as native

ROOT = Path("/home/bangtliu/coord/logs")


def oracle(x, ids, w, ws):
    valid = (ids.flatten() >= 0) & (ids.flatten() < 64)
    slots = torch.arange(64, device=x.device)[valid]
    experts = ids.flatten()[valid]
    token = slots // 8
    rows = []
    for expert in experts.unique().tolist():
        use = experts == expert
        idx = slots[use]
        xx = x[token[use]].double()
        dense = native.dequant_mxfp4(w["w13"][expert], w["w13_scale"][expert]).double()
        gate, up = (xx @ dense.T).split(4096, -1)
        ref = (gate.sigmoid() * gate * up).bfloat16()
        dense2 = native.dequant_mxfp4(w["w2"][expert], w["w2_scale"][expert]).double()
        matched = (ws.inter[idx].double() @ dense2.T).bfloat16()
        full = (ref.double() @ dense2.T).bfloat16()
        row = dict(
            expert=expert,
            inter=error(ws.inter[idx], ref),
            w2=error(ws.output[idx], matched),
            end=error(ws.output[idx], full),
        )
        assert row["inter"]["relative_l2"] < 0.01 and row["w2"]["relative_l2"] < 0.01
        rows.append(row)
    return rows


def single_eight(device):
    ids = torch.full((8, 8), 255, device=device, dtype=torch.int32)
    ids[:, 0] = 3
    return ids


def native_extra():
    assert triton.__version__.split(".")[:2] == ["3", "9"]
    for seed in SEEDS:
        x, w = load(seed)
        ids = single_eight("cuda")
        table = native.moe_align_block_size_small(ids, 64, 0, 16)
        inter = torch.full(
            (64, 4096), float("nan"), device=x.device, dtype=torch.bfloat16
        )
        out = torch.full_like(inter, float("nan"))
        cfg = native._pick_config(8, 8, 64, True)

        def call():
            native._launch_gemm(
                x,
                w["w13"],
                w["w13_scale"],
                inter,
                *table,
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
                out,
                *table,
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
        record = dict(inter=inter.cpu(), y=out.cpu(), time=graph_times(call))
        torch.save(record, ROOT / f"phase3e1b-native-one8-{seed}.pt")
        print(json.dumps(dict(seed=seed, case="one8", time=record["time"])), flush=True)


def main():
    require_gpu0()
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-reference", action="store_true")
    if parser.parse_args().native_reference:
        native_extra()
        return
    records = []
    for seed in SEEDS:
        x, w = load(seed)
        refs = torch.load(ROOT / f"phase3e1-native-{seed}.pt", weights_only=True)
        ws = stage.make_workspace(x.device)
        old_ws = baseline.make_workspace(x.device)
        refs["one8"] = torch.load(
            ROOT / f"phase3e1b-native-one8-{seed}.pt", weights_only=True
        )
        patterns = cases(seed, "cuda")
        patterns["one8"] = single_eight("cuda")
        for name, ids in patterns.items():
            table = native.moe_align_block_size_small(ids, 64, 0, 16)
            ws.inter[:64].fill_(float("nan"))
            ws.output.fill_(float("nan"))
            call = lambda: stage.experts(
                x, *table, w["w13"], w["w13_scale"], w["w2"], w["w2_scale"], ws
            )
            k = call()
            torch.cuda.synchronize()
            assert ws.status.eq(1).all()
            assert k.n_spills == 0
            valid = ids.flatten() < 64
            ref = refs[name]
            comparison = None
            if valid.any():
                comparison = dict(
                    inter=error(ws.inter[:64][valid], ref["inter"].cuda()[valid]),
                    output=error(ws.output[valid], ref["y"].cuda()[valid]),
                )
                assert (
                    comparison["inter"]["relative_l2"] < 0.01
                    and comparison["output"]["relative_l2"] < 0.01
                )
            baseline.experts(
                x, ids, w["w13"], w["w13_scale"], w["w2"], w["w2_scale"], old_ws
            )
            torch.cuda.synchronize()
            assert old_ws.status.eq(1).all()
            hit = torch.arange(64, device=x.device)[valid]
            e = ids.flatten()[valid].long()
            m = hit // 8
            assert torch.equal(ws.inter[:64][valid], old_ws.inter[e, m])
            assert torch.equal(ws.output[valid], old_ws.output[e, m])
            assert ws.inter[64].eq(0).all()
            assert (
                ws.inter[:64][~valid].isnan().all() and ws.output[~valid].isnan().all()
            )
            fp64 = oracle(x, ids, w, ws)
            timing = graph_times(call)
            assert ws.status.eq(1).all()
            unique = ids.flatten()[valid].unique().numel()
            nbytes = unique * 26738688
            read = None
            if nbytes:
                data = torch.zeros(nbytes, device="cuda", dtype=torch.uint8)
                sink = torch.empty(
                    (nbytes + 4095) // 4096, device="cuda", dtype=torch.int32
                )
                read = graph_times(
                    lambda: stream_read[(sink.numel(),)](data, sink, nbytes, 4096)
                )
                del data, sink
            blocks = table[2].item() // 16
            record = dict(
                seed=seed,
                case=name,
                hits=int(valid.sum()),
                distinct=unique,
                m_blocks=blocks,
                w13_tasks=blocks * 64,
                w2_tasks=blocks * 64,
                ctas=192,
                registers=k.n_regs,
                spills=k.n_spills,
                shared=k.metadata.shared,
                stage=timing,
                read=read,
                native39=ref["time"],
                native_error=comparison,
                oracle=fp64,
                baseline_bitwise=True,
            )
            print(json.dumps(record), flush=True)
            records.append(record)
            if name == "typical":
                (ROOT / f"phase3e1b-integrated-{seed}.amdgcn").write_text(
                    k.asm["amdgcn"]
                )
        # Missing arrival last for this workspace; stale tag from prior successful launch.
        ids = patterns["one"]
        table = native.moe_align_block_size_small(ids, 64, 0, 16)
        stage.experts(x, *table, w["w13"], w["w13_scale"], w["w2"], w["w2_scale"], ws)
        torch.cuda.synchronize()
        old = ws.flags.clone()
        ws.output.fill_(float("nan"))
        stage.experts(
            x, *table, w["w13"], w["w13_scale"], w["w2"], w["w2_scale"], ws, skip=0
        )
        torch.cuda.synchronize()
        failed = int(ws.status.eq(0).sum())
        assert failed == 64
        # M-block0 never publishes its new tag; all16 of those original slots stay unwritten.
        first = table[0][:16].long()
        assert ws.output[first].isnan().all()
        assert torch.equal(ws.flags[0], old[0])
        print(
            json.dumps(
                dict(
                    event="missing",
                    seed=seed,
                    failed=failed,
                    absent_w13=1,
                    stale_rejected=True,
                )
            ),
            flush=True,
        )
    (ROOT / "phase3e1b-results.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

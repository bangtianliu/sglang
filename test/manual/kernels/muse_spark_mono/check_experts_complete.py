"""GPU0-only complete S5: native/FP64, changing delayed replays, bounded missing CTA."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from bench_hbm import graph_times
from check_experts import cases
from check_experts_routed import oracle, single_eight
from check_experts_valu import mixed
from layer_fixture import require_gpu0
from reference_archive import checked_load

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    experts_complete as stage,
)
from sglang.srt.rocm_muse_spark import muse_moe_triton as native

ROOT = Path("/home/bangtliu/coord/logs")


def error(a, b):
    a, b = a.double(), b.double()
    return dict(
        max_abs=float((a - b).abs().max()),
        relative_l2=float((a - b).norm() / b.norm().clamp_min(1e-30)),
    )


def table_for(ids):
    tables = [native.moe_align_block_size_small(ids, 64, r * 64, 16) for r in range(4)]
    return tuple(
        torch.stack([t[i] for t in tables]).contiguous().reshape(-1) for i in range(3)
    )


def reference(x, ids, scores, w, gamma, table):
    inter = torch.full(
        (4, 64, 4096), float("nan"), device=x.device, dtype=torch.bfloat16
    )
    y = torch.full_like(inter, float("nan"))
    partial = torch.empty((4, 8, 4096), device=x.device, dtype=torch.bfloat16)
    cfg = native._pick_config(8, 8, 64, True)
    t = [v.reshape(4, -1) for v in table]

    def call():
        for r in range(4):
            native._launch_gemm(
                x,
                w["w13"][r],
                w["w13_scale"][r],
                inter[r],
                t[0][r],
                t[1][r],
                t[2][r],
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
                inter[r],
                w["w2"][r],
                w["w2_scale"][r],
                y[r],
                t[0][r],
                t[1][r],
                t[2][r],
                4096,
                4096,
                64,
                8,
                False,
                False,
                True,
                cfg,
            )
            native.finalize_norm_weighted_sum_kernel[(8,)](
                y[r],
                gamma,
                scores,
                ids,
                partial[r],
                4096,
                r * 64,
                64,
                1e-8,
                1.0,
                0,
                TOP_K=8,
                HAS_NORM=True,
                HAS_SCALE=True,
                BLOCK_H=4096,
                SPLIT_K=1,
                num_warps=4,
                enable_fp_fusion=False,
            )

    call()
    reduced = torch.zeros((8, 4096), device=x.device, dtype=torch.float32)
    for r in range(4):
        reduced = reduced + partial[r].float()
    return dict(
        inter=inter,
        y=y,
        partial=partial,
        reduced=reduced,
        time=graph_times(call, samples=30),
    )


def matched(ids, scores, gamma, y):
    effective = (gamma.float() + 1).to(gamma.dtype).double()
    weighted = torch.zeros((4, 8, 4096), device=y.device, dtype=torch.float64)
    for r in range(4):
        for m in range(8):
            for s in range(8):
                e = int(ids[m, s])
                if r * 64 <= e < (r + 1) * 64:
                    value = (
                        (y[r, m * 8 + s].float() * effective.float())
                        .bfloat16()
                        .double()
                    )
                    inv = torch.rsqrt(value.square().sum() / 4096 + 1e-8)
                    weighted[r, m] += scores[m, s].double() * inv * value
    return weighted


def main():
    require_gpu0()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--case", choices=("typical", "one", "one8", "distinct", "empty", "mixed")
    )
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()
    saved = checked_load(f"phase3e1-inputs-{args.seed}.pt")
    x = saved["x"]
    # Distinct virtual-rank weights from deterministic expert permutations.
    weights = {
        k: torch.stack([v.roll(r * 11, 0) for r in range(4)])
        for k, v in saved["weights"].items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    gamma = (torch.randn(4096, generator=generator) * 0.02).to(
        device="cuda", dtype=torch.bfloat16 if args.seed != 7 else torch.float32
    )
    scores = torch.rand((8, 8), generator=generator).cuda()
    scores /= scores.sum(1, keepdim=True)
    patterns = cases(args.seed, "cuda")
    patterns.update(one8=single_eight("cuda"), mixed=mixed("cuda"))
    for name in ("one8", "mixed", "empty"):
        patterns[name] = torch.where(patterns[name] == 255, 256, patterns[name])
    delay = torch.tensor([args.seed], device="cuda", dtype=torch.int32)
    ids = patterns[args.case or "typical"]
    table = table_for(ids)
    k = 9
    while True:
        ws = stage.make_workspace(x.device, k=k)
        compiled = []
        for skip, seed in ((-1, None), (-1, delay), (0, None)):
            kernel = stage.s5(
                x,
                ids,
                scores,
                table,
                weights,
                gamma,
                ws,
                skip=skip,
                delay_seed=seed,
                compile_only=True,
            )
            resource = kernel_occupancy(
                f"S5-skip{skip}-delay{seed is not None}", kernel
            )
            row = dict(
                grid=ws.grid,
                vgprs=kernel.n_regs,
                lds=kernel.metadata.shared,
                spills=kernel.n_spills,
                occupancy=resource.active_ctas_per_cu,
            )
            compiled.append(row)
            print(json.dumps(dict(event="compiled", **row)), flush=True)
        limit = min(r["occupancy"] for r in compiled)
        if k <= limit:
            break
        assert limit > 0
        k = limit
    records = []
    for name, ids in patterns.items():
        if args.case and name != args.case:
            continue
        table = table_for(ids)
        ws.inter.fill_(float("nan"))
        ws.output.fill_(float("nan"))
        ws.reduced.fill_(float("nan"))
        ref = reference(x, ids, scores, weights, gamma, table)
        call = lambda: stage.s5(x, ids, scores, table, weights, gamma, ws)
        kernel = call()
        torch.cuda.synchronize()
        print(
            json.dumps(
                dict(
                    event="status",
                    case=name,
                    passed=int(ws.status[: ws.grid].sum()),
                    grid=ws.grid,
                )
            ),
            flush=True,
        )
        assert ws.status[: ws.grid].eq(1).all(), "bounded S5 pipeline failed"
        errors = []
        oracles = []
        for r in range(4):
            valid = (ids.flatten() >= r * 64) & (ids.flatten() < (r + 1) * 64)
            assert (
                ws.inter[r, ~valid].isnan().all() and ws.output[r, ~valid].isnan().all()
            )
            if valid.any():
                e = dict(
                    inter=error(ws.inter[r, valid], ref["inter"][r, valid]),
                    output=error(ws.output[r, valid], ref["y"][r, valid]),
                )
                assert (
                    e["inter"]["relative_l2"] < 0.01
                    and e["output"]["relative_l2"] < 0.015
                ), e
                errors.append(e)
                oracles.extend(
                    oracle(
                        x,
                        ids - r * 64,
                        {n: v[r] for n, v in weights.items()},
                        SimpleNamespace(inter=ws.inter[r], output=ws.output[r]),
                    )
                )
        ideal = matched(ids, scores, gamma, ws.output)
        matched_error = error(ws.weighted, ideal)
        assert matched_error["relative_l2"] < 2e-6, matched_error
        summed = torch.zeros((8, 4096), device=x.device)
        for r in range(4):
            summed = summed + ws.partial[r].float()
        assert torch.equal(ws.reduced, summed.expand_as(ws.reduced)), (
            "rank order exactness"
        )
        final_error = error(ws.reduced, ref["reduced"].expand_as(ws.reduced))
        assert final_error["relative_l2"] < 0.02, final_error
        record = dict(
            seed=args.seed,
            case=name,
            grid=ws.grid,
            k=k,
            compiled=compiled,
            native_errors=errors,
            oracle=oracles,
            matched_weighted=matched_error,
            native_final=final_error,
            time=graph_times(call, samples=30),
            native_serial_four_ranks=ref["time"],
        )
        print(json.dumps(record), flush=True)
        records.append(record)
        (ROOT / f"phase3e2-results-{args.seed}.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        if name == "typical" or args.case:
            for fmt in ("amdgcn", "ttgir"):
                (ROOT / f"phase3e2-complete-{args.seed}.{fmt}").write_text(
                    kernel.asm[fmt]
                )
    # Alternate immutable reference fixtures while changing inputs, rank placement, scores and delays.
    if args.replays:
        base_ids = torch.where(single_eight("cuda") == 255, 256, single_eight("cuda"))
        variants = []
        for i in range(2):
            xx = x if i == 0 else -x
            ii = (
                base_ids
                if i == 0
                else torch.where(base_ids < 64, base_ids + 128, base_ids)
            )
            sc = scores if i == 0 else scores * 0.75
            tt = table_for(ii)
            rr = reference(xx, ii, sc, weights, gamma, tt)["reduced"]
            variants.append((xx, ii, sc, tt, rr))
        for launch in range(args.replays):
            xx, ii, sc, tt, rr = variants[launch % 2]
            delay.fill_(args.seed + launch)
            stage.s5(xx, ii, sc, tt, weights, gamma, ws, delay_seed=delay)
            torch.cuda.synchronize()
            assert ws.status[: ws.grid].eq(1).all(), ("replay-timeout", launch)
            assert error(ws.reduced, rr.expand_as(ws.reduced))["relative_l2"] < 0.02, (
                "replay",
                launch,
            )
        print(
            json.dumps(
                dict(
                    event="delayed-changing-replays", launches=args.replays, passed=True
                )
            ),
            flush=True,
        )
    ws.reduced.fill_(float("nan"))
    stage.s5(x, ids, scores, table, weights, gamma, ws, skip=0)
    torch.cuda.synchronize()
    print(
        json.dumps(
            dict(
                event="missing-producer",
                ready=int(ws.status[: ws.grid].sum()),
                grid=ws.grid,
            )
        ),
        flush=True,
    )
    assert ws.status[: ws.grid].eq(0).all() and ws.reduced.isnan().all()


if __name__ == "__main__":
    main()

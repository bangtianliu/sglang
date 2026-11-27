"""Common per-rank S5: native/FP64, exact peer sum, changing replays and timeouts."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from check_experts import cases
from check_experts_complete import error, matched, reference, table_for
from check_experts_routed import oracle, single_eight
from check_experts_valu import mixed
from layer_fixture import require_gpu0
from rank_timing import rank_times
from reference_archive import checked_load
from stage_replay import verify_pair

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_experts as stage,
)

ROOT = Path("/home/bangtliu/coord/logs")


def native_rank_time(x, ids, scores, table, weights, gamma, rank):
    from bench_hbm import graph_times

    from sglang.srt.rocm_muse_spark import muse_moe_triton as native

    inter = torch.empty((64, 4096), device=x.device, dtype=torch.bfloat16)
    y = torch.empty_like(inter)
    partial = torch.empty_like(x)
    cfg = native._pick_config(8, 8, 64, True)

    def call():
        native._launch_gemm(
            x,
            weights["w13"],
            weights["w13_scale"],
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
            weights["w2"],
            weights["w2_scale"],
            y,
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
        native.finalize_norm_weighted_sum_kernel[(8,)](
            y,
            gamma,
            scores,
            ids,
            partial,
            4096,
            rank * 64,
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

    return graph_times(call, samples=30)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--case")
    p.add_argument("--rank", type=int)
    p.add_argument("--replays", type=int, default=200)
    p.add_argument("--read-mode", type=int, default=2)
    p.add_argument("--w2-sk", type=int, default=1)
    a = p.parse_args()
    original_s5 = stage.s5

    def selected_s5(*args, **kw):
        return original_s5(*args, **kw, read_mode=a.read_mode, w2_sk=a.w2_sk)

    stage.s5 = selected_s5
    saved = checked_load(f"phase3e1-inputs-{a.seed}.pt")
    x = saved["x"]
    w = {
        k: torch.stack([v.roll(r * 11, 0) for r in range(4)])
        for k, v in saved["weights"].items()
    }
    gen = torch.Generator().manual_seed(a.seed)
    gamma = (torch.randn(4096, generator=gen) * 0.02).to(
        device="cuda", dtype=torch.float32 if a.seed == 7 else torch.bfloat16
    )
    scores = torch.rand((8, 8), generator=gen).cuda()
    scores /= scores.sum(1, keepdim=True)
    patterns = cases(a.seed, "cuda")
    patterns.update(one8=single_eight("cuda"), mixed=mixed("cuda"))
    for name in ("one8", "mixed", "empty"):
        patterns[name] = torch.where(patterns[name] == 255, 256, patterns[name])
    seed = torch.tensor([a.seed], device="cuda", dtype=torch.int32)
    rows = []
    ranks = range(4) if a.rank is None else (a.rank,)
    for name, ids in patterns.items():
        if a.case and a.case != name:
            continue
        tables = table_for(ids)
        ref = reference(x, ids, scores, w, gamma, tables)
        for rank in ranks:
            ws = stage.make_workspace("cuda")
            weights = {k: v[rank] for k, v in w.items()}
            table = tuple(t.reshape(4, -1)[rank] for t in tables)
            ws.inter.fill_(float("nan"))
            ws.output.fill_(float("nan"))
            setup = lambda: standin.prepare(
                ws.peers, ref["partial"], ws.state.epochs, rank=rank
            )
            call = lambda: stage.s5(
                x, ids, scores, table, weights, gamma, ws, rank=rank
            )
            setup()
            kernel = call()
            torch.cuda.synchronize()
            print(
                json.dumps(
                    dict(
                        event="status",
                        case=name,
                        rank=rank,
                        ready=int(ws.state.status.sum()),
                        resources=list(ws.state.admitted.values()),
                    )
                ),
                flush=True,
            )
            assert ws.state.status.eq(1).all()
            valid = (ids.flatten() >= rank * 64) & (ids.flatten() < (rank + 1) * 64)
            assert (
                ws.inter[rank, ~valid].isnan().all()
                and ws.output[rank, ~valid].isnan().all()
            )
            for other in range(4):
                if other != rank:
                    assert (
                        ws.inter[other].isnan().all() and ws.output[other].isnan().all()
                    )
            errors = {}
            if valid.any():
                errors = dict(
                    inter=error(ws.inter[rank, valid], ref["inter"][rank, valid]),
                    output=error(ws.output[rank, valid], ref["y"][rank, valid]),
                )
                assert (
                    errors["inter"]["relative_l2"] < 0.01
                    and errors["output"]["relative_l2"] < 0.015
                ), errors
            primary = oracle(
                x,
                ids - rank * 64,
                weights,
                SimpleNamespace(inter=ws.inter[rank], output=ws.output[rank]),
            )
            ideal = matched(ids, scores, gamma, ws.output)[rank]
            weighted = error(ws.weighted[rank], ideal)
            assert weighted["relative_l2"] < 2e-6, weighted
            summed = torch.zeros_like(ref["reduced"])
            for r in range(4):
                summed = (
                    summed + (ws.partial[r] if r == rank else ref["partial"][r]).float()
                )
            assert torch.equal(ws.peers.reduced[rank], summed)
            final = error(ws.peers.reduced[rank], ref["reduced"])
            assert final["relative_l2"] < 0.02, final
            timing = rank_times(setup, call)
            row = dict(
                seed=a.seed,
                read_mode=a.read_mode,
                w2_sk=a.w2_sk,
                case=name,
                rank=rank,
                native=errors,
                fp64=primary,
                weighted=weighted,
                final=final,
                timing=timing,
                resources=list(ws.state.admitted.values()),
                native_serial_four_ranks=ref["time"],
                native_one_rank_without_collective=native_rank_time(
                    x, ids, scores, table, weights, gamma, rank
                ),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            (ROOT / f"phase40b-S5-results-{a.seed}-{a.case or 'all'}.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
            if name == "typical":
                for fmt in ("amdgcn", "ttgir"):
                    (ROOT / f"phase40b-S5-r{rank}-{a.seed}.{fmt}").write_text(
                        kernel.asm[fmt]
                    )
    if a.replays:
        for rank in ranks:
            ws = stage.make_workspace("cuda")
            weights = {k: v[rank] for k, v in w.items()}
            ii = single_eight("cuda")
            ii = torch.where(ii < 64, ii + rank * 64, 256)
            tt = table_for(ii)
            table = tuple(v.reshape(4, -1)[rank] for v in tt)
            variants = []
            for variant, sign in enumerate((1, -1, -1, 1)):
                xx = x * sign
                route_ids = (
                    torch.full_like(ii, 256)
                    if variant == 1
                    else torch.where(ii < 256, ii + 14, ii)
                    if variant == 3
                    else ii
                )
                route_scores = scores * (0.75 if variant == 3 else 1.0)
                route_tables = table_for(route_ids)
                route_table = tuple(v.reshape(4, -1)[rank] for v in route_tables)
                rr = reference(xx, route_ids, route_scores, w, gamma, route_tables)
                standin.prepare(ws.peers, rr["partial"], ws.state.epochs, rank=rank)
                stage.s5(
                    xx,
                    route_ids,
                    route_scores,
                    route_table,
                    weights,
                    gamma,
                    ws,
                    rank=rank,
                    delay_seed=seed,
                )
                torch.cuda.synchronize()
                assert ws.state.status.eq(1).all()
                assert (
                    error(ws.peers.reduced[rank], rr["reduced"])["relative_l2"] < 0.02
                )
                variants.append(
                    (
                        xx,
                        route_ids,
                        route_scores,
                        route_table,
                        rr["partial"],
                        ws.peers.reduced[rank].clone(),
                    )
                )
            assert not torch.equal(variants[0][5], variants[1][5])
            chunks = 128
            errors = torch.full(
                (a.replays, chunks), -1, device="cuda", dtype=torch.int32
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(a.replays):
                    xx, route_ids, route_scores, route_table, peer, expected = variants[
                        i % len(variants)
                    ]
                    seed.fill_(a.seed + i)
                    standin.prepare(ws.peers, peer, ws.state.epochs, rank=rank)
                    stage.s5(
                        xx,
                        route_ids,
                        route_scores,
                        route_table,
                        weights,
                        gamma,
                        ws,
                        rank=rank,
                        delay_seed=seed,
                    )
                    verify_pair[(chunks,)](
                        ws.peers.reduced[rank],
                        ws.peers.reduced[rank],
                        expected,
                        expected,
                        ws.state.status,
                        errors,
                        i,
                        8 * 4096,
                        ws.state.grid,
                        chunks,
                        num_warps=4,
                    )
            graph.replay()
            torch.cuda.synchronize()
            assert errors.eq(0).all(), int(errors.sum())
            (ROOT / f"phase40b-S5-replays-{a.seed}-r{rank}.json").write_text(
                json.dumps(errors.cpu().tolist()) + "\n"
            )
            print(
                json.dumps(
                    dict(event="replay", rank=rank, launches=a.replays, mismatches=0)
                ),
                flush=True,
            )
            ws.peers.reduced.fill_(float("nan"))
            standin.prepare(ws.peers, variants[0][4], ws.state.epochs, rank=rank)
            stage.s5(
                variants[0][0], ii, scores, table, weights, gamma, ws, rank=rank, skip=0
            )
            torch.cuda.synchronize()
            assert not ws.state.status.eq(1).all() and ws.peers.reduced.isnan().all()
            print(
                json.dumps(
                    dict(
                        event="missing",
                        rank=rank,
                        ready=int(ws.state.status.sum()),
                        rejected=True,
                        terminal_nan=True,
                    )
                ),
                flush=True,
            )
    print("PASS common S5", flush=True)


if __name__ == "__main__":
    main()

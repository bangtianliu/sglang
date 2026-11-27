"""Per-rank S3 native/FP64 suite with stand-in peers, replays and missing producer."""

import argparse
import json
from pathlib import Path

import torch
from check_tail import check, oracle_boundary
from layer_fixture import require_gpu0
from rank_timing import rank_times
from reference_archive import checked_load
from stage_replay import verify_pair

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attn_tail as stage,
)

ROOT = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rank", type=int)
    p.add_argument("--replays", type=int, default=200)
    a = p.parse_args()
    d = checked_load(f"S3-inputs-{a.seed}.pt")
    native = checked_load(f"S3-native-{a.seed}.pt")
    rows = []
    for rank in range(4) if a.rank is None else (a.rank,):
        ws = stage.make_workspace("cuda")
        x = d["x"][rank].clone()
        gate = d["gate"][rank]
        weight = d["weight"][rank]
        original = x.clone()
        seed = torch.tensor([a.seed], device="cuda", dtype=torch.int32)
        out = torch.empty_like(d["stream"], dtype=torch.bfloat16)
        updated = torch.empty_like(d["stream"])

        def setup(sign):
            standin.prepare(
                ws.peers, native[sign]["partial"], ws.state.epochs, rank=rank
            )

        def call(skip=-1, delayed=True):
            return stage.attention_tail(
                x,
                gate,
                weight,
                d["stream"],
                d["pre"],
                d["alpha"],
                d["beta"],
                rank=rank,
                workspace=ws,
                delay_seed=seed if delayed else None,
                skip=skip,
                output=(out, updated),
            )

        baselines = []
        sign_results = []
        for sign in (1, -1):
            x.copy_(original * sign)
            setup(sign)
            actual = call()
            torch.cuda.synchronize()
            print(
                json.dumps(
                    dict(
                        event="status",
                        rank=rank,
                        sign=sign,
                        ready=int(ws.state.status.sum()),
                        resources=list(ws.state.admitted.values()),
                    )
                ),
                flush=True,
            )
            assert ws.state.status.eq(1).all()
            raw = x.double().reshape(8, 32, 64)
            ng = (
                (
                    raw
                    * (raw.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
                    * gate.double().reshape_as(raw).sigmoid()
                )
                .reshape(8, 2048)
                .bfloat16()
            )
            fp64 = (ng.double() @ weight.double().T).bfloat16()
            local = ws.partial[rank]
            match = torch.zeros_like(d["stream"])
            ideal = torch.zeros_like(d["stream"])
            for r in range(4):
                match = (
                    match + (local if r == rank else native[sign]["partial"][r]).float()
                )
                ideal = (
                    ideal + (fp64 if r == rank else native[sign]["partial"][r]).float()
                )
            assert torch.equal(ws.peers.reduced[rank], match)
            expected = oracle_boundary(
                match.bfloat16(), d["stream"], d["pre"], d["alpha"], d["beta"]
            )
            primary = oracle_boundary(
                ideal.bfloat16(), d["stream"], d["pre"], d["alpha"], d["beta"]
            )
            sign_results.append(
                dict(
                    sign=sign,
                    partial_native=check(local, native[sign]["partial"][rank]),
                    partial_fp64=check(local, fp64),
                    native=[
                        check(v, n) for v, n in zip(actual, native[sign]["output"])
                    ],
                    matched_fp64=[check(v, n) for v, n in zip(actual, expected)],
                    fp64=[check(v, n) for v, n in zip(actual, primary)],
                )
            )
            baselines.append(tuple(v.clone() for v in actual))
        x.copy_(original)
        timing = rank_times(lambda: setup(1), lambda: call(delayed=False))
        timing["cta_delays_excluded"] = True
        assert ws.state.status.eq(1).all()
        for got, want in zip((out, updated), baselines[0]):
            assert torch.equal(got, want)
        chunks = out.numel() // 256
        errors = torch.full((a.replays, chunks), -1, device="cuda", dtype=torch.int32)
        if a.replays:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(a.replays):
                    sign = 1 if i % 2 == 0 else -1
                    x.copy_(original if sign == 1 else -original)
                    seed.fill_(a.seed + i)
                    setup(sign)
                    call()
                    verify_pair[(chunks,)](
                        out,
                        updated,
                        *baselines[i % 2],
                        ws.state.status,
                        errors,
                        i,
                        out.numel(),
                        ws.state.grid,
                        chunks,
                        num_warps=4,
                    )
            graph.replay()
            torch.cuda.synchronize()
            assert errors.eq(0).all(), int(errors.sum())
            assert ws.state.status.eq(1).all()
            (ROOT / f"phase40b-S3-replays-{a.seed}-r{rank}.json").write_text(
                json.dumps(errors.cpu().tolist()) + "\n"
            )
        out.fill_(float("nan"))
        updated.fill_(float("nan"))
        ws.state.status.fill_(-1)
        setup(1)
        call(skip=0)
        torch.cuda.synchronize()
        assert (
            (not ws.state.status.eq(1).all())
            and out.isnan().all()
            and updated.isnan().all()
        )
        row = dict(
            rank=rank,
            seed=a.seed,
            signs=sign_results,
            timing=timing,
            resources=list(ws.state.admitted.values()),
            replays=a.replays,
            device_mismatches=int(errors.sum()),
            missing_ready=int(ws.state.status.sum()),
            standin_setup_excluded=True,
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
        (ROOT / f"phase40b-S3-results-{a.seed}.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
    print("PASS common per-rank S3", flush=True)


if __name__ == "__main__":
    main()

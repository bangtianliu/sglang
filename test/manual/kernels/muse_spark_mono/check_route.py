"""S4 native joint projection/top8/slot-table parity, deliberate ties and replay."""

import argparse
import json
from pathlib import Path

import torch
from bench_hbm import graph_times, stream_read
from check_tail import check
from layer_fixture import build_layer, require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.ops.moe.router_pooler_topk import (
    _joint_finish,
    joint_router_pooler_topk,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.route import (
    _standalone,
    make_workspace,
    routing,
)
from sglang.srt.rocm_muse_spark.muse_moe_triton import moe_align_block_size_small

LOG = Path("/home/bangtliu/coord/logs")
REFS = LOG


def oracle(x, weight, bias, norm):
    p = x.double() @ weight.double().T
    raw = ((p[:, 4096:4352] + p[:, 4352:]) * (1 / 32**0.5)).sigmoid()
    ids = torch.argsort(raw + bias.double(), descending=True, stable=True)[:, :8]
    selected = torch.gather(raw, 1, ids)
    scores = selected / (selected.sum(1, keepdim=True) + 1e-15)
    pool = p[:, :4096].bfloat16().double()
    pool = (
        pool * (pool.square().mean(1, keepdim=True) + 1e-5).rsqrt() * norm.double()
    ).bfloat16()
    return scores, ids.int(), pool


def native(x, weight, bias, norm, rank):
    scores, ids, pool = joint_router_pooler_topk(
        x, weight, bias, norm, 1 / 32**0.5, 1.0, 1e-5
    )
    return scores, ids, pool, *moe_align_block_size_small(ids, 64, rank * 64, 16)


@g.jit
def verify(
    X,
    Expected,
    Status,
    Errors,
    STEP: gl.constexpr,
    WHICH: gl.constexpr,
    N: gl.constexpr,
):
    i = gl.program_id(0) * 256 + gl.arange(
        0, 256, layout=gl.BlockedLayout([1], [32], [4], [0])
    )
    wrong = (
        gl.load(X + i, i < N, other=0) != gl.load(Expected + i, i < N, other=0)
    ) | (gl.load(Status + i % 192) != 1)
    gl.store(
        Errors + STEP * 768 + WHICH * 128 + gl.program_id(0),
        gl.sum(wrong.to(gl.int32), 0),
    )


def replay(x, weight, bias, norm, rank, ws, references):
    original = x.clone()
    delay = torch.tensor([7], device=x.device, dtype=torch.int32)

    def call():
        return routing(x, weight, bias, norm, rank=rank, workspace=ws, delay_seed=delay)

    baselines = []
    for sign in (1, -1):
        x.copy_(original * sign)
        actual = call()
        torch.cuda.synchronize()
        expected = references[sign][rank]
        for i, (a, b) in enumerate(zip(actual, expected)):
            if i in (0, 2):
                check(a, b)
            else:
                assert torch.equal(a, b), i
        baselines.append(tuple(a.clone() for a in actual))
    assert not torch.equal(baselines[0][2], baselines[1][2])
    records = torch.full((200, 6, 128), -1, device=x.device, dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for step in range(200):
            x.copy_(original if step % 2 == 0 else -original)
            delay.fill_(20261117 + step)
            actual = call()
            for which, (a, b) in enumerate(zip(actual, baselines[step % 2])):
                verify[(128,)](
                    a, b, ws.state.status, records, step, which, a.numel(), num_warps=4
                )
    graph.replay()
    torch.cuda.synchronize()
    assert records.eq(0).all(), records.sum().item()
    (LOG / "phase3d-S4-replay-records.json").write_text(
        json.dumps(records.cpu().tolist()) + "\n"
    )
    print(
        json.dumps(
            dict(event="replay", stage="S4", launches=200, tensors=6, mismatches=0)
        ),
        flush=True,
    )
    # Missing projection producer last; no router/pooler output can be accepted.
    s = ws.state
    s.status.fill_(-1)
    _standalone[(192,)](
        x,
        weight,
        bias,
        norm,
        ws.projection,
        s.epochs,
        s.arrivals,
        s.flags,
        s.status,
        ws.scores,
        ws.ids,
        ws.pooled,
        ws.sorted_ids,
        ws.expert_ids,
        ws.num_post,
        rank * 64,
        SKIP=143,
        num_warps=16,
        enable_fp_fusion=False,
    )
    torch.cuda.synchronize()
    expected_status = torch.zeros_like(s.status)
    expected_status[143] = -1
    assert torch.equal(s.status, expected_status), s.status.tolist()
    print(
        json.dumps(
            dict(event="missing_arrival", stage="S4", skipped=143, failed=191, absent=1)
        ),
        flush=True,
    )


def main():
    global REFS
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--saved-references", action="store_true")
    parser.add_argument("--native-reference", action="store_true")
    args = parser.parse_args()
    if args.saved_references:
        from reference_archive import ROOT, verify

        verify("S4")
        REFS = ROOT
    require_gpu0()
    if args.prepare:
        for seed in (7, 20261117):
            f = build_layer(seed=seed, components=("router", "pooler", "norms"))
            values = dict(
                x=f.x,
                weight=f.weights["joint_router_pooler"],
                bias=f.weights["expert_bias"],
                norm=f.weights["pre_expert_effective"],
            )
            torch.save(
                {k: v.cpu() for k, v in values.items()},
                REFS / f"phase3d-S4-inputs-{seed}.pt",
            )
        return
    if args.native_reference:
        import triton

        for seed in (7, 20261117):
            base = torch.load(REFS / f"phase3d-S4-inputs-{seed}.pt", weights_only=True)
            for tied in (False, True):
                data = {k: v.cuda() for k, v in base.items()}
                if tied:
                    data["weight"][4096:].zero_()
                    data["bias"].zero_()
                    data["bias"][64:80] = 0.25
                x, weight, bias, norm = (
                    data[k] for k in ("x", "weight", "bias", "norm")
                )
                values = {}
                for sign in (1, -1):
                    values[sign] = []
                    oracle_result = oracle(x * sign, weight, bias, norm)
                    for rank in range(4):
                        result = native(x * sign, weight, bias, norm, rank)
                        assert torch.equal(result[1], oracle_result[1])
                        check(result[0], oracle_result[0], 1e-4)
                        check(result[2], oracle_result[2])
                        values[sign].append(tuple(t.cpu() for t in result))
                timing = graph_times(lambda: native(x, weight, bias, norm, 0))
                values["timing"] = timing
                values["stack"] = dict(
                    torch=str(torch.__version__), triton=str(triton.__version__)
                )
                torch.save(values, LOG / f"phase3d-S4-native-{seed}-ties{tied}.pt")
                print(
                    json.dumps(
                        dict(
                            event="native_reference",
                            seed=seed,
                            ties=tied,
                            timing=timing,
                            **values["stack"],
                        )
                    ),
                    flush=True,
                )
        return
    results = []
    replay_args = None
    for seed in (7, 20261117):
        from types import SimpleNamespace

        data = {
            k: v.cuda()
            for k, v in torch.load(
                REFS / f"phase3d-S4-inputs-{seed}.pt", weights_only=True
            ).items()
        }
        f = SimpleNamespace(x=data["x"])
        weight, bias, norm = data["weight"], data["bias"], data["norm"]
        for tied in (False, True):
            if tied:
                weight = weight.clone()
                weight[4096:].zero_()
                bias = torch.zeros_like(bias)
                bias[64:80] = 0.25
            saved = torch.load(
                REFS / f"phase3d-S4-native-{seed}-ties{tied}.pt", weights_only=True
            )
            references = {
                sign: [tuple(t.cuda() for t in row) for row in saved[sign]]
                for sign in (1, -1)
            }
            ref = oracle(f.x, weight, bias, norm)
            for rank in range(4):
                ws = make_workspace(f.x.device)
                call = lambda: routing(f.x, weight, bias, norm, rank=rank, workspace=ws)
                actual = call()
                torch.cuda.synchronize()
                assert ws.state.status.eq(1).all(), ws.state.status.tolist()
                own = references[1][rank]
                if not (
                    torch.equal(actual[1], own[1]) and torch.equal(actual[1], ref[1])
                ):
                    expected_projection = f.x.double() @ weight.double().T
                    diff = (
                        ws.projection.view(torch.float32).double() - expected_projection
                    ).abs()
                    print(
                        json.dumps(
                            dict(
                                event="ids_mismatch",
                                actual=actual[1].tolist(),
                                native=own[1].tolist(),
                                oracle=ref[1].tolist(),
                                projection_max=diff.max().item(),
                                router_high_max=diff[:, 4096:4352].max().item(),
                                router_low_max=diff[:, 4352:].max().item(),
                            )
                        ),
                        flush=True,
                    )
                assert torch.equal(actual[1], own[1]) and torch.equal(
                    actual[1], ref[1]
                ), "top8 ids"
                for i in (3, 4, 5):
                    assert torch.equal(actual[i], own[i]), ("slot table", i, rank)
                if tied:
                    assert torch.equal(
                        actual[1], torch.arange(64, 72, device="cuda").expand(8, -1)
                    )
                # Native epilogue on exactly our completed FP32 projection isolates rounding.
                matched = _joint_finish(
                    ws.projection.view(torch.float32)[None, :, :],
                    1,
                    8,
                    bias,
                    norm,
                    1 / 32**0.5,
                    1.0,
                    1e-5,
                    None,
                )
                assert torch.equal(actual[1], matched[1])
                row = dict(
                    stage="S4",
                    seed=seed,
                    rank=rank,
                    deliberate_ties=tied,
                    scores_vs_native=check(actual[0], own[0], 1e-4),
                    pooled_vs_native=check(actual[2], own[2]),
                    scores_vs_fp64=check(actual[0], ref[0], 1e-4),
                    pooled_vs_fp64=check(actual[2], ref[2]),
                    matched_scores=check(actual[0], matched[0], 1e-5),
                    matched_pool=check(actual[2], matched[2], 0.004),
                    ids_exact=True,
                    slots_exact=True,
                    num_post=actual[5].item(),
                )
                if seed == 7 and not tied and rank == 0:
                    row["timing"] = graph_times(call)
                    row["native"] = saved["timing"]
                    row["native_stack"] = saved["stack"]
                    source = torch.rand((weight.numel() // 2,), device="cuda")
                    sink = torch.empty((source.numel() // 4096,), device="cuda")
                    row["read"] = graph_times(
                        lambda: stream_read[(sink.numel(),)](
                            source, sink, source.numel(), 4096
                        )
                    )
                    row["weight_bytes"] = weight.numel() * 2
                print(json.dumps(row), flush=True)
                results.append(row)
                (LOG / "phase3d-S4-results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
            if seed == 20261117 and not tied:
                replay_args = (
                    f.x,
                    weight,
                    bias,
                    norm,
                    3,
                    make_workspace(f.x.device),
                    references,
                )
    replay(*replay_args)
    print("PASS16 S4 cases", flush=True)


if __name__ == "__main__":
    main()

"""S3/S6 full correctness, delayed replays and faults using hashed3.8 native refs."""

import argparse
import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from check_tail import check, check_boundary_rounding, oracle_boundary
from layer_fixture import require_gpu0
from reference_archive import checked_load
from stage_replay import exercise

LOG = Path("/home/bangtliu/coord/logs")


def s3(seed):
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attn_tail import (
        _standalone,
        attention_tail,
        make_workspace,
    )

    d = checked_load(f"S3-inputs-{seed}.pt")
    native = checked_load(f"S3-native-{seed}.pt")
    x, gate, weight, stream, pre, alpha, beta = (
        d[k] for k in ("x", "gate", "weight", "stream", "pre", "alpha", "beta")
    )
    original = x.clone()
    ws = make_workspace(x.device)
    call = lambda: attention_tail(
        x, gate, weight, stream, pre, alpha, beta, workspace=ws
    )
    actual = call()
    torch.cuda.synchronize()
    assert ws.state.status.eq(1).all()
    partial = []
    for r in range(4):
        raw = x[r].double().reshape(8, 32, 64)
        ng = (
            (
                raw
                * (raw.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
                * gate[r].double().reshape_as(raw).sigmoid()
            )
            .reshape(8, 2048)
            .bfloat16()
        )
        partial.append((ng.double() @ weight[r].double().T).bfloat16())
    oracle_sum = torch.zeros_like(stream)
    exact_sum = torch.zeros_like(stream)
    for r in range(4):
        oracle_sum += partial[r].float()
        exact_sum += ws.partial[r].float()
    assert torch.equal(ws.reduced, exact_sum.expand_as(ws.reduced))
    oracle = oracle_boundary(oracle_sum.bfloat16(), stream, pre, alpha, beta)
    matched = oracle_boundary(exact_sum.bfloat16(), stream, pre, alpha, beta)
    row = dict(
        stage="S3",
        seed=seed,
        partial_vs_native=check(ws.partial, native[1]["partial"]),
        partial_vs_fp64=check(ws.partial, torch.stack(partial)),
        vs_native=[
            check(a, b.expand_as(a)) for a, b in zip(actual, native[1]["output"])
        ],
        vs_fp64=[check(a, b.expand_as(a)) for a, b in zip(actual, oracle)],
        matched_fp64=[check(a, b.expand_as(a)) for a, b in zip(actual, matched)],
        timing=graph_times(call),
    )
    out, updated = torch.empty_like(actual[0]), torch.empty_like(actual[1])
    state = ws.state
    args = (
        x,
        gate,
        weight,
        stream,
        pre,
        alpha,
        beta,
        ws.partial,
        ws.reduced,
        state.epochs,
        state.arrivals,
        state.flags,
        state.status,
        ws.scratch,
        ws.peers,
        ws.ready,
        out,
        updated,
        ws.payload,
        ws.flags,
    )
    _standalone[(128,)](*args, ROTATE=17, num_warps=16, enable_fp_fusion=False)
    torch.cuda.synchronize()
    assert state.status.eq(1).all()
    assert torch.equal(out, actual[0]) and torch.equal(updated, actual[1])
    delay = torch.tensor([seed], device=x.device, dtype=torch.int32)
    delayed = lambda: attention_tail(
        x, gate, weight, stream, pre, alpha, beta, workspace=ws, delay_seed=delay
    )
    reference = lambda: tuple(
        t.expand(4, -1, -1)
        for t in native[1 if torch.equal(x, original) else -1]["output"]
    )
    row["replay"] = exercise(delayed, x, state, delay, reference, f"3e1f0-S3-{seed}")
    state.status.fill_(-1)
    out.fill_(float("nan"))
    updated.fill_(float("nan"))
    _standalone[(128,)](*args, SKIP=127, num_warps=16, enable_fp_fusion=False)
    torch.cuda.synchronize()
    assert state.status[:127].eq(0).all() and state.status[127] == -1
    assert out.isnan().all() and updated.isnan().all()
    row["missing"] = dict(failed=127, absent=1, output_unwritten=True)
    print(json.dumps(row), flush=True)
    return [row]


def s6(seed):
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tail import (
        _standalone,
        make_workspace,
        unpool_boundary,
    )

    if seed == 7:
        pattern = checked_load("S6-boundary-patterns.pt")

        def pattern_reference(branch, stream, pre, alpha, beta, post=None, last=False):
            assert all(
                torch.equal(a, pattern[k])
                for a, k in (
                    (branch, "branch"),
                    (stream, "stream"),
                    (alpha, "alpha"),
                    (beta, "beta"),
                    (post, "post"),
                )
            )
            if pre is not None:
                assert torch.equal(pre, pattern["pre"])
            return pattern["outputs"][f"{pre is not None}-{last}"]

        check_boundary_rounding(pattern_reference)
    d = checked_load(f"S6-inputs-{seed}.pt")
    native = checked_load(f"S6-native-{seed}.pt")
    x, weight, stream, post, alpha, beta = (
        d[k] for k in ("x", "weight", "stream", "post", "alpha", "beta")
    )
    original = x.clone()
    projection = (x.double() @ weight.double().T).bfloat16()
    rows = []
    for kind in ("normal", "last_weighted", "last_unweighted"):
        pre = None if kind == "last_unweighted" else d["pre"]
        last = kind != "normal"
        ws = make_workspace(x.device)
        call = lambda: unpool_boundary(
            x, weight, stream, post, pre, alpha, beta, pre_bf16=last, workspace=ws
        )
        actual = call()
        torch.cuda.synchronize()
        assert ws.state.status.eq(1).all()
        oracle = oracle_boundary(
            projection, stream, pre, alpha, beta, post=post, last=last
        )
        matched = oracle_boundary(
            ws.projection.view(torch.float32).bfloat16(),
            stream,
            pre,
            alpha,
            beta,
            post=post,
            last=last,
        )
        row = dict(
            stage="S6",
            seed=seed,
            kind=kind,
            projection_vs_native=check(
                ws.projection.view(torch.float32), native[1]["projection"]
            ),
            projection_vs_fp64=check(ws.projection.view(torch.float32), projection),
            vs_native=[
                check(a, b) for a, b in zip(actual, native[1]["modes"][kind]["output"])
            ],
            vs_fp64=[check(a, b) for a, b in zip(actual, oracle)],
            matched_fp64=[check(a, b) for a, b in zip(actual, matched)],
            timing=graph_times(call),
        )
        if kind == "normal":
            normal = tuple(t.clone() for t in actual)
        else:
            assert torch.equal(actual[1], normal[1])
            assert not torch.equal(actual[0], normal[0])
        changing = x.clone()
        delay = torch.tensor([seed], device=x.device, dtype=torch.int32)
        delayed = lambda: unpool_boundary(
            changing,
            weight,
            stream,
            post,
            pre,
            alpha,
            beta,
            pre_bf16=last,
            workspace=ws,
            delay_seed=delay,
        )
        reference = lambda: native[1 if torch.equal(changing, original) else -1][
            "modes"
        ][kind]["output"]
        row["replay"] = exercise(
            delayed, changing, ws.state, delay, reference, f"3e1f0-S6-{seed}-{kind}"
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Missing producer/stale flags last, with a fresh normal-mode workspace.
    ws = make_workspace(x.device)
    state = ws.state
    unpool_boundary(x, weight, stream, post, d["pre"], alpha, beta, workspace=ws)
    old = state.flags[0, 0, 0].item()
    state.status.fill_(-1)
    out = torch.full((8, 8192), float("nan"), device=x.device, dtype=torch.bfloat16)
    updated = torch.full((8, 8192), float("nan"), device=x.device)
    _standalone[(192,)](
        x,
        weight,
        stream,
        post,
        d["pre"],
        alpha,
        beta,
        ws.projection,
        state.epochs,
        state.arrivals,
        state.flags,
        state.status,
        out,
        updated,
        False,
        SKIP=127,
        num_warps=16,
        enable_fp_fusion=False,
    )
    torch.cuda.synchronize()
    want = torch.zeros_like(state.status)
    want[127] = -1
    assert (
        torch.equal(state.status, want) and out.isnan().all() and updated.isnan().all()
    )
    assert state.flags[0, 0, 0] == old
    print(
        json.dumps(dict(event="S6_missing", failed=191, absent=1, stale_rejected=True)),
        flush=True,
    )
    return rows


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("S3", "S6"), required=True)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rows = s3(a.seed) if a.stage == "S3" else s6(a.seed)
    (LOG / f"phase3e1f0-{a.stage}-{a.seed}.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

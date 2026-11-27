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


def s6(seed):
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.common_tail import (
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

        check_boundary_rounding(pattern_reference, waves=4)
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
            delayed, changing, ws.state, delay, reference, f"4-S6-{seed}-{kind}"
        )
        row["resources"] = list(ws.state.admitted.values())
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Missing producer/stale flags last, with a fresh normal-mode workspace.
    ws = make_workspace(x.device)
    state = ws.state
    unpool_boundary(x, weight, stream, post, d["pre"], alpha, beta, workspace=ws)
    out = torch.full((8, 8192), float("nan"), device=x.device, dtype=torch.bfloat16)
    updated = torch.full((8, 8192), float("nan"), device=x.device)
    unpool_boundary(
        x,
        weight,
        stream,
        post,
        d["pre"],
        alpha,
        beta,
        workspace=ws,
        skip=127,
        output=(out, updated),
    )
    torch.cuda.synchronize()
    assert not state.status.eq(1).all() and out.isnan().all() and updated.isnan().all()
    print(
        json.dumps(
            dict(
                event="S6_missing",
                ready=int(state.status.sum()),
                rejected=True,
                stale_rejected=True,
            )
        ),
        flush=True,
    )

    return rows


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rows = s6(a.seed)
    (LOG / f"phase40b-S6-{a.seed}.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

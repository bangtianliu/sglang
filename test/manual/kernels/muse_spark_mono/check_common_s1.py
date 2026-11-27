"""Reuse S1 native/FP64/cache cases and delayed replay checks on the common shape."""

import argparse
import json
from pathlib import Path

import check_s1 as suite
import torch
from check_s1_replay import verify
from layer_fixture import EPS, Q_SCALE, require_gpu0
from reference_archive import fixture

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_qkvg as stage,
)

ROOT = Path("/home/bangtliu/coord/logs")


def replays(rope):
    f = fixture(20261117)
    ws = stage.make_workspace(f.x.device)
    seed = torch.tensor([7], device=f.x.device, dtype=torch.int32)
    original = f.x.clone()

    def call(**opts):
        return stage.qkvg(
            f.x,
            f.weights["qkvg"],
            f.positions,
            f.cos_sin if rope else None,
            f.cache_k,
            f.cache_v,
            f.locations,
            f.mapping if rope else None,
            EPS,
            Q_SCALE,
            f.k_scale,
            f.v_scale,
            workspace=ws,
            delay_seed=seed,
            **opts,
        )

    actual = call()
    torch.cuda.synchronize()
    expected, caches_ref = suite.fp64(f, rope=rope, mapping=rope, round_stages=True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.double(), b, atol=0.125, rtol=0.02)
    baseline = [v.clone() for v in actual]
    slots = f.mapping[f.locations] if rope else f.locations
    caches = [
        v.view(torch.uint8).reshape(-1, 256)[slots].clone().view(torch.float8_e4m3fn)
        for v in (f.cache_k, f.cache_v)
    ]
    records = torch.full((1000, 6), -1, device=f.x.device, dtype=torch.int32)
    # Compile/validate both sign branches before graph capture.
    for sign in (-1, 1):
        verify.warmup(
            *actual,
            f.cache_k,
            f.cache_v,
            *baseline,
            *caches,
            slots,
            records[0],
            sign,
            grid=(6,),
            num_warps=8,
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(1000):
            sign = 1 if i % 2 == 0 else -1
            f.x.copy_(original if sign == 1 else -original)
            seed.fill_(20261117 + i)
            actual = call()
            verify[(6,)](
                *actual,
                f.cache_k,
                f.cache_v,
                *baseline,
                *caches,
                slots,
                records[i],
                sign,
                num_warps=8,
            )
    graph.replay()
    torch.cuda.synchronize()
    assert records.eq(0).all(), int(records.sum())
    assert ws.status.eq(1).all() and ws.state[-72:].eq(ws.state[0]).all()
    (ROOT / f"phase40b-S1-replay-records-{rope}.json").write_text(
        json.dumps(records.cpu().tolist()) + "\n"
    )
    print(
        json.dumps(
            dict(
                event="replays",
                rope=rope,
                launches=1000,
                mismatches=0,
                resources=list(ws.admitted.values()),
            )
        ),
        flush=True,
    )
    before = [v.view(torch.uint8).clone() for v in (f.cache_k, f.cache_v)]
    ws.status.fill_(-1)
    ws.state[-72:].fill_(99)
    call(skip=0)
    torch.cuda.synchronize()
    assert ws.status[0] == 0 and ws.status[1:].eq(1).all()
    assert ws.state[-72] != ws.state[0] and ws.state[-71:].eq(ws.state[0]).all()
    assert all(
        torch.equal(v.view(torch.uint8), b)
        for v, b in zip((f.cache_k, f.cache_v), before)
    )
    print(
        json.dumps(
            dict(
                event="missing",
                rope=rope,
                ready=int(ws.status.sum()),
                stale_rejected=True,
            )
        ),
        flush=True,
    )


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--replays-only", action="store_true")
    args = p.parse_args()
    suite.ARCHIVE = True
    resources = []

    def common_call(*a, **kw):
        result = stage.qkvg(*a, **kw)
        resources.extend(kw["workspace"].admitted.values())
        return result

    suite.qkvg = common_call
    suite.make_workspace = stage.make_workspace
    rows = []
    if not args.replays_only:
        for context in (8192,) if args.smoke else (8192, 32768, 131072):
            for seed in (7,) if args.smoke else (7, 20261117):
                for rope, mapping in ((False, False), (True, True)):
                    rows.append(
                        suite.one_case(
                            context=context,
                            seed=seed,
                            rope=rope,
                            mapping=mapping,
                            ctas=768,
                            head_tagged=True,
                        )
                    )
        if not args.smoke:
            for rope, mapping in (
                (False, False),
                (True, True),
                (True, False),
                (False, True),
            ):
                suite.check_epilogue(rope=rope, mapping=mapping, waves=4)
            for rope, mapping in ((True, True), (True, False), (False, True)):
                rows.append(
                    suite.one_case(
                        context=8192,
                        seed=17,
                        rope=rope,
                        mapping=mapping,
                        ctas=768,
                        head_tagged=True,
                        skip_slots=True,
                    )
                )
        (ROOT / "phase40b-S1-results.json").write_text(
            json.dumps(dict(cases=rows, resources=resources), indent=2) + "\n"
        )
    if not args.smoke:
        for rope in (False, True):
            replays(rope)
    print("PASS common S1 native/FP64/cache/replay suite", flush=True)


if __name__ == "__main__":
    main()

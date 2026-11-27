"""S2 native/FP64/page-map suite, delayed replays and missing arrivals at k4."""

import argparse
import json
from pathlib import Path

import check_s2 as suite
import torch
from check_s2_replay import verify
from layer_fixture import build_layer, require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attention as stage,
)
from sglang.test.kernels.tlx_attention import paged_reference

ROOT = Path("/home/bangtliu/coord/logs")


def replays(window):
    f = build_layer(context=131072, components=())
    torch.manual_seed(20261117)
    q = torch.randn((8, 32, 64), device="cuda", dtype=torch.bfloat16)
    original = q.clone()
    ws = stage.make_workspace(q.device, context=131072, window_left=window)
    f.seq_lens.fill_(131067)
    tail = f.page_table.flatten()[-1].item()
    f.cache_v.view(torch.uint8)[tail, 11:] = 127
    k, v, mapping = f.cache_k, f.cache_v, None
    if window >= 0:
        first = (131067 - 8 - window) // 16 * 16
        logical = torch.arange(first, 131072, device="cuda")
        source = f.page_table.flatten()[logical // 16].long() * 16 + logical % 16
        allocation = torch.randperm(logical.numel(), device="cuda")
        mapping = torch.full((131072,), -1, dtype=torch.int64, device="cuda")
        mapping[source] = allocation
        kb = torch.empty((logical.numel(), 4, 64), dtype=torch.uint8, device="cuda")
        vb = torch.empty_like(kb)
        kb[allocation] = f.cache_k.view(torch.uint8).reshape(-1, 4, 64)[source]
        vb[allocation] = f.cache_v.view(torch.uint8).reshape(-1, 4, 64)[source]
        k = kb.view(torch.float8_e4m3fn).reshape(-1, 16, 4, 64)
        v = vb.view(torch.float8_e4m3fn).reshape(-1, 16, 4, 64)
    seed = torch.tensor([7], device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)

    def run(skip=-1):
        return stage.attention(
            q,
            k,
            v,
            f.page_table,
            f.seq_lens,
            f.k_scale,
            f.v_scale,
            context=131072,
            window_left=window,
            mapping=mapping,
            workspace=ws,
            delay_seed=seed,
            skip=skip,
            output=output,
        )

    args = dict(
        q=q,
        k=f.cache_k,
        v=f.cache_v,
        block_table=f.page_table,
        k_descale=f.k_scale,
        v_descale=f.v_scale,
        softmax_scale=0.125,
        causal=True,
        window_size=(window, 0),
    )
    baselines = []
    for sign in (1, -1):
        q.copy_(original * sign)
        run()
        torch.cuda.synchronize()
        assert ws.status.eq(1).all() and ws.cta_status.eq(1).all()
        baselines.append(output.clone())
        args["q"] = original if sign == 1 else -original
        expected = paged_reference(args, [8], [131067])
        error = float((output.double() - expected).abs().max() / expected.abs().max())
        print(
            json.dumps(
                dict(event="tail-oracle", window=window, sign=sign, relative_linf=error)
            ),
            flush=True,
        )
        assert torch.isfinite(output).all() and error < 0.008, error
    errors = torch.full((200, 64), -1, device="cuda", dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(200):
            q.copy_(original if i % 2 == 0 else -original)
            seed.fill_(20261117 + i)
            run()
            verify[(64,)](output, baselines[i % 2], errors, i, num_warps=4)
    graph.replay()
    torch.cuda.synchronize()
    assert errors.eq(0).all() and ws.status.eq(1).all() and ws.cta_status.eq(1).all()
    (ROOT / f"phase4-S2-replay-records-{window}.json").write_text(
        json.dumps(errors.cpu().tolist()) + "\n"
    )
    print(
        json.dumps(
            dict(
                event="replays",
                window=window,
                length=131067,
                nan_tail=True,
                launches=200,
                mismatches=0,
                resources=list(ws.admitted.values()),
            )
        ),
        flush=True,
    )
    output.fill_(float("nan"))
    ws.status.fill_(-1)
    ws.cta_status.fill_(-1)
    previous = ws.flags[0, 0].item()
    run(skip=0)
    torch.cuda.synchronize()
    assert ws.status.eq(0).all() and ws.cta_status.eq(0).all() and output.isnan().all()
    assert ws.flags[0, 0].item() == previous
    print(
        json.dumps(
            dict(
                event="missing",
                window=window,
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
    a = p.parse_args()
    resources = []

    def call(*args, **kw):
        out = stage.attention(*args, **kw)
        resources.extend(kw["workspace"].admitted.values())
        return out

    suite.attention = call
    suite.make_workspace = stage.make_workspace
    rows = []
    if not a.replays_only:
        for ctx in (8192,) if a.smoke else (8192, 32768, 131072):
            for seed in (7,) if a.smoke else (7, 20261117):
                for left in (-1, 2047):
                    rows.append(suite.case(ctx, left, seed, ctas=768, waves=4))
        if not a.smoke:
            for ctx in (8192, 32768, 131072):
                rows.append(suite.case(ctx, 2047, 7, ctas=768, waves=4, page_map=True))
        (ROOT / "phase4-S2-results.json").write_text(
            json.dumps(dict(cases=rows, resources=resources), indent=2) + "\n"
        )
    if not a.smoke:
        for window in (-1, 2047):
            replays(window)
    print("PASS common S2 suite", flush=True)


if __name__ == "__main__":
    main()

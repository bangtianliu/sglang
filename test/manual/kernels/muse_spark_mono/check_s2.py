"""S2 causal/window/page/mapping and split-merge oracle checks on GPU0."""

import argparse
import json
from pathlib import Path

import torch
from bench_hbm import graph_times, stream_read
from layer_fixture import build_layer, require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attention import (
    attention,
    make_workspace,
)
from sglang.test.kernels.tlx_attention import paged_reference

LOG = Path("/home/bangtliu/coord/logs")


def case(context=8192, window=-1, seed=7, ctas=192, bn=64, waves=8, page_map=False):
    page_map = page_map and window >= 0
    f = build_layer(seed=seed, context=context, components=())
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    q = (
        torch.randn((8, 32, 64), device="cuda", dtype=torch.bfloat16, generator=gen) * 2
    ).contiguous()
    # Sliding cache holds only the aligned union of all eight query windows.
    # Token-granular permutation catches accidental page-only map handling.
    mapping = None
    k, v = f.cache_k, f.cache_v
    if window >= 0:
        first = max(0, context - 8 - window) // 16 * 16
        logical = torch.arange(first, context, device="cuda")
        pages = f.page_table.flatten()[logical // 16].long()
        source = pages * 16 + logical % 16
        count = logical.numel()
        allocation = torch.randperm(count, device="cuda", generator=gen).to(torch.int64)
        if page_map:
            allocation = (
                torch.randperm(count // 16, device="cuda", generator=gen)[:, None] * 16
                + torch.arange(16, device="cuda")[None, :]
            ).flatten()
        mapping = torch.full((context,), -1, device="cuda", dtype=torch.int64)
        mapping[source] = allocation
        k = torch.empty((count, 4, 64), device="cuda", dtype=torch.uint8)
        v = torch.empty_like(k)
        k[allocation] = f.cache_k.view(torch.uint8).reshape(context, 4, 64)[source]
        v[allocation] = f.cache_v.view(torch.uint8).reshape(context, 4, 64)[source]
        k = k.view(torch.float8_e4m3fn).reshape(count // 16, 16, 4, 64)
        v = v.view(torch.float8_e4m3fn).reshape(count // 16, 16, 4, 64)
    ws = make_workspace(
        q.device, context=context, window_left=window, ctas=ctas, block_n=bn
    )
    fn = lambda: attention(
        q,
        k,
        v,
        f.page_table,
        f.seq_lens,
        f.k_scale,
        f.v_scale,
        context=context,
        window_left=window,
        mapping=mapping,
        mapping_page_aligned=page_map,
        workspace=ws,
        ctas=ctas,
        block_n=bn,
        waves=waves,
    )
    out = fn()
    torch.cuda.synchronize()
    assert ws.status.eq(1).all(), ws.status.tolist()
    args = dict(
        q=q,
        k=f.cache_k,
        v=f.cache_v,
        out=torch.empty_like(q),
        cu_seqlens_q=f.query_start,
        seqused_k=f.seq_lens,
        max_seqlen_q=8,
        max_seqlen_k=context,
        softmax_scale=0.125,
        causal=True,
        window_size=(window, 0),
        block_table=f.page_table,
        k_descale=f.k_scale,
        v_descale=f.v_scale,
    )
    oracle = paged_reference(args, [8], [context])
    diff = (out.double() - oracle).abs()
    rel = (diff.max() / oracle.abs().max()).item()
    assert torch.isfinite(out).all()
    assert rel < 0.008, rel
    result = dict(
        context=context,
        window=window,
        seed=seed,
        ctas=ctas,
        bn=bn,
        waves=waves,
        splits=ws.splits,
        page_map=page_map,
        max_abs=diff.max().item(),
        rel_linf=rel,
        timing=graph_times(fn),
    )
    assert ws.status.eq(1).all()
    # Native comparison uses the equivalent original cache/table before slot remapping.
    try:
        from aiter.ops.triton.attention.unified_attention import unified_attention
    except ImportError as e:
        result["aiter_unavailable"] = str(e)
    else:
        native = lambda: unified_attention(**args, softcap=0.0, q_descale=None)
        actual = native()
        if actual is None:
            actual = args["out"]
        delta = (actual.double() - oracle).abs()
        assert delta.max() / oracle.abs().max() < 0.008
        result["aiter"] = dict(
            max_abs=delta.max().item(),
            candidate_max_abs=(out.float() - actual.float()).abs().max().item(),
            timing=graph_times(native),
        )
    nbytes = (context if window < 0 else min(context, window + 8)) * 4 * 64 * 2
    # Same byte count and timing helper as the candidate; zero pad to FP32 size.
    n = (nbytes + 3) // 4
    x = torch.rand((n,), device="cuda")
    y = torch.empty(((n + 4095) // 4096,), device="cuda")
    read = lambda: stream_read[((n + 4095) // 4096,)](x, y, n, 4096)
    read()
    expected = torch.nn.functional.pad(x, (0, (-n) % 4096)).reshape(-1, 4096).sum(1)
    torch.testing.assert_close(y, expected)
    result["read"] = graph_times(read)
    result["kv_bytes"] = nbytes
    result["read_ratio"] = result["timing"]["median_us"] / result["read"]["median_us"]
    print(json.dumps(result), flush=True)
    return result


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--window", type=int, default=-1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ctas", type=int, default=192)
    p.add_argument("--bn", type=int, default=64)
    p.add_argument("--waves", type=int, default=8)
    p.add_argument("--all", action="store_true")
    p.add_argument("--page-map", action="store_true")
    args = p.parse_args()
    results = []
    for context, window, seed in (
        [
            (n, w, s)
            for n in (8192, 32768, 131072)
            for w in (-1, 2047)
            for s in (7, 20261117)
        ]
        if args.all
        else [(args.context, args.window, args.seed)]
    ):
        results.append(
            case(context, window, seed, args.ctas, args.bn, args.waves, args.page_map)
        )
        (LOG / "phase3c-S2-opt-results.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()

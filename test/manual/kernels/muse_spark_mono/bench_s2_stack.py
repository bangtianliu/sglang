"""TP4 full/sliding comparison on identical saved tensors across toolchain images."""

import argparse
import json
from pathlib import Path

import torch
import triton
from bench_hbm import graph_times
from layer_fixture import build_layer, require_gpu0

from sglang.test.kernels.tlx_attention import paged_reference

LOG = Path("/home/bangtliu/coord/logs")
INPUT = LOG / "phase3c-S2-opt2-shared-inputs.pt"


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--impl", choices=("stage", "utlx", "aiter"))
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if args.prepare:
        assert not INPUT.exists(), INPUT
        f = build_layer(context=131072, seed=7, components=())
        gen = torch.Generator(device="cuda").manual_seed(8)
        q = (
            torch.randn((8, 32, 64), device="cuda", dtype=torch.bfloat16, generator=gen)
            * 2
        )
        torch.save(
            dict(
                q=q.cpu(),
                k=f.cache_k.cpu(),
                v=f.cache_v.cpu(),
                block_table=f.page_table.cpu(),
                seqused_k=f.seq_lens.cpu(),
                cu_seqlens_q=f.query_start.cpu(),
                k_descale=f.k_scale.cpu(),
                v_descale=f.v_scale.cpu(),
            ),
            INPUT,
        )
        print("saved", INPUT, flush=True)
        return
    assert args.impl and args.tag
    data = {
        k: v.to("cuda:0")
        for k, v in torch.load(INPUT, map_location="cpu", weights_only=True).items()
    }
    if args.impl == "utlx":
        from sglang.kernels.ops.attention.tlx.paged import unified_attention
    elif args.impl == "aiter":
        from aiter.ops.triton.attention.unified_attention import unified_attention
    else:
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attention import (
            attention,
            make_workspace,
        )
    results = []
    for window in (-1, 2047):
        out = torch.empty_like(data["q"])
        kw = dict(
            **data,
            out=out,
            max_seqlen_q=8,
            max_seqlen_k=131072,
            softmax_scale=0.125,
            causal=True,
            window_size=(window, 0),
            q_descale=None,
            softcap=0.0,
        )
        if args.impl == "stage":
            ws = make_workspace(out.device, context=131072, window_left=window)
            call = lambda: attention(
                data["q"],
                data["k"],
                data["v"],
                data["block_table"],
                data["seqused_k"],
                data["k_descale"],
                data["v_descale"],
                context=131072,
                window_left=window,
                workspace=ws,
            )
        else:
            call = lambda: unified_attention(**kw)
        actual = call()
        torch.cuda.synchronize()
        if actual is None:
            actual = out
        expected = paged_reference(kw, [8], [131072])
        diff = (actual.double() - expected).abs()
        rel = (diff.max() / expected.abs().max()).item()
        assert torch.isfinite(actual).all() and rel < 0.008, (window, rel)
        timing = graph_times(call)
        if args.impl == "stage":
            assert ws.status.eq(1).all()
        row = dict(
            impl=args.impl,
            tag=args.tag,
            context=131072,
            q_shape=list(data["q"].shape),
            kv_shape=list(data["k"].shape),
            window=window,
            max_abs=diff.max().item(),
            rel_linf=rel,
            timing=timing,
            torch=torch.__version__,
            triton=triton.__version__,
            input=str(INPUT),
            device=torch.cuda.get_device_properties(0).gcnArchName,
        )
        print(json.dumps(row), flush=True)
        results.append(row)
        (LOG / f"phase3c-S2-opt2-stack-{args.impl}-{args.tag}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()

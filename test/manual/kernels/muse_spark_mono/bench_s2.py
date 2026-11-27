"""All-head S2 partial-only and fused timing, with FP64 checks before timing."""

import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from layer_fixture import build_layer, require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attention import (
    _partial,
    _standalone,
    attention,
    make_workspace,
)
from sglang.test.kernels.tlx_attention import paged_reference


@g.jit
def partial_only(
    Q,
    K,
    V,
    PAGES,
    LENGTH,
    KS,
    VS,
    OUT,
    LSE,
    SPLITS: gl.constexpr,
    BN: gl.constexpr,
    MAX_PAGES: gl.constexpr,
    PREFETCH: gl.constexpr,
):
    task = gl.program_id(0)
    value, lse = _partial(
        Q,
        K,
        V,
        PAGES,
        None,
        LENGTH,
        KS,
        VS,
        task,
        0.125,
        -1,
        SPLITS,
        BN,
        False,
        MAX_PAGES,
        False,
        PREFETCH,
    )
    layout: gl.constexpr = value.type.layout
    h = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    r = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    gl.store(
        OUT
        + task * 16384
        + h[:, None, None] * 4096
        + r[None, :, None] * 64
        + d[None, None, :],
        value,
    )
    gl.store(LSE + task * 256 + h[:, None] * 64 + r[None, :], lse)


def main():
    require_gpu0()
    f = build_layer(context=131072, components=())
    q = torch.randn((8, 32, 64), device="cuda", dtype=torch.bfloat16)
    expected = paged_reference(
        dict(
            q=q,
            k=f.cache_k,
            v=f.cache_v,
            block_table=f.page_table,
            k_descale=f.k_scale,
            v_descale=f.v_scale,
            softmax_scale=0.125,
            causal=True,
            window_size=(-1, 0),
        ),
        [8],
        [131072],
    )
    for bn in (32, 64, 128):
        for waves in (8, 16):
            for prefetch in (2, 3):
                ws = make_workspace(q.device, context=131072, block_n=bn)
                out = torch.empty((ws.splits, 4, 64, 64), device="cuda")
                lse = torch.empty((ws.splits, 4, 64), device="cuda")
                fn = lambda: partial_only[(ws.splits,)](
                    q,
                    f.cache_k,
                    f.cache_v,
                    f.page_table,
                    f.seq_lens,
                    f.k_scale,
                    f.v_scale,
                    out,
                    lse,
                    ws.splits,
                    bn,
                    ws.pages_per_chunk,
                    prefetch,
                    num_warps=waves,
                    enable_fp_fusion=False,
                )
                kernel = fn()
                name = f"phase3c-S2-opt-bn{bn}-w{waves}-p{prefetch}"
                Path("/home/bangtliu/coord/logs/" + name + ".amdgcn").write_text(
                    kernel.asm["amdgcn"]
                )
                fused = lambda: attention(
                    q,
                    f.cache_k,
                    f.cache_v,
                    f.page_table,
                    f.seq_lens,
                    f.k_scale,
                    f.v_scale,
                    context=131072,
                    workspace=ws,
                    block_n=bn,
                    waves=waves,
                    prefetch=prefetch,
                )
                actual = fused()
                rel = (
                    (actual.double() - expected).abs().max() / expected.abs().max()
                ).item()
                assert ws.status.eq(1).all() and rel < 0.008, rel
                print(
                    json.dumps(
                        dict(
                            bn=bn,
                            waves=waves,
                            prefetch=prefetch,
                            partial=graph_times(fn),
                            fused=graph_times(fused),
                            rel_linf=rel,
                            metadata=str(kernel.metadata),
                        )
                    ),
                    flush=True,
                )
                kernel = _standalone[(192,)](
                    q,
                    f.cache_k,
                    f.cache_v,
                    f.page_table,
                    None,
                    f.seq_lens,
                    f.k_scale,
                    f.v_scale,
                    ws.partial,
                    ws.lse,
                    ws.counters,
                    ws.flags,
                    ws.status,
                    torch.empty_like(q),
                    0.125,
                    -1,
                    ws.splits,
                    bn,
                    False,
                    ws.pages_per_chunk,
                    False,
                    prefetch,
                    num_warps=waves,
                    enable_fp_fusion=False,
                )
                Path("/home/bangtliu/coord/logs/" + name + "-fused.amdgcn").write_text(
                    kernel.asm["amdgcn"]
                )


if __name__ == "__main__":
    main()

"""S2 changing-input replay, device metadata, and missing-arrival bounds."""

import argparse
import json
from pathlib import Path

import torch
from layer_fixture import build_layer, require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attention import (
    attention,
    attention_stage,
    make_workspace,
)
from sglang.test.kernels.tlx_attention import paged_reference


@g.jit
def delayed(
    Q,
    K,
    V,
    PAGES,
    MAP,
    LENGTH,
    KS,
    VS,
    P,
    LSE,
    COUNTERS,
    FLAGS,
    STATUS,
    OUT,
    SEED,
    SPLITS: gl.constexpr,
    MAX_PAGES: gl.constexpr,
    SKIP: gl.constexpr = -1,
    LEFT: gl.constexpr = -1,
    HAS_MAP: gl.constexpr = False,
):
    pid = gl.program_id(0)
    if pid != SKIP:
        seed = gl.load(SEED)
        delay = ((pid * 1103515245 + seed) ^ ((pid + seed) >> 3)) & 31
        for _ in range(delay):
            sync.backoff()
        attention_stage(
            pid,
            gl.num_programs(0),
            Q,
            K,
            V,
            PAGES,
            MAP,
            LENGTH,
            KS,
            VS,
            P,
            LSE,
            COUNTERS,
            FLAGS,
            STATUS,
            OUT,
            0.125,
            LEFT,
            SPLITS,
            64,
            HAS_MAP,
            MAX_PAGES,
        )


@g.jit
def verify(X, EXPECTED, ERROR, STEP: gl.constexpr):
    i = gl.program_id(0) * 256 + gl.arange(
        0, 256, layout=gl.BlockedLayout([1], [32], [4], [0])
    )
    x = gl.load(X + i)
    e = gl.load(EXPECTED + i)
    mismatch = gl.sum((x != e).to(gl.int32), 0)
    gl.store(ERROR + STEP * 64 + gl.program_id(0), mismatch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, choices=(-1, 2047), default=-1)
    window = parser.parse_args().window
    require_gpu0()
    f = build_layer(context=131072, components=())
    torch.manual_seed(20261117)
    q = torch.randn((8, 32, 64), device="cuda", dtype=torch.bfloat16)
    original = q.clone()
    ws = make_workspace(q.device, context=131072, window_left=window)
    # Non-page-aligned device length, with a NaN-filled unused physical tail.
    f.seq_lens.fill_(131067)
    tail = f.page_table.flatten()[-1].item()
    f.cache_v.view(torch.uint8)[tail, 11:] = 127
    k, v, mapping = f.cache_k, f.cache_v, None
    if window >= 0:
        # Retain the unused NaN tail in a compact, token-permuted SWA pool.
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
    out = attention(
        q,
        k,
        v,
        f.page_table,
        f.seq_lens,
        f.k_scale,
        f.v_scale,
        context=131072,
        workspace=ws,
        window_left=window,
        mapping=mapping,
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
    expected = paged_reference(args, [8], [131067])
    assert torch.isfinite(out).all()
    assert (out.double() - expected).abs().max() / expected.abs().max() < 0.008
    assert ws.status.eq(1).all()
    seed = torch.tensor(7, device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)

    def run(skip=-1):
        delayed[(192,)](
            q,
            k,
            v,
            f.page_table,
            mapping,
            f.seq_lens,
            f.k_scale,
            f.v_scale,
            ws.partial,
            ws.lse,
            ws.counters,
            ws.flags,
            ws.status,
            output,
            seed,
            ws.splits,
            ws.pages_per_chunk,
            skip,
            window,
            mapping is not None,
            num_warps=8,
            enable_fp_fusion=False,
        )

    baselines = []
    for sign in (1, -1):
        q.copy_(original * sign)
        run()
        torch.cuda.synchronize()
        assert ws.status.eq(1).all()
        baselines.append(output.clone())
    for i, b in enumerate(baselines):
        args["q"] = original if i == 0 else -original
        expected = paged_reference(args, [8], [131067])
        assert (b.double() - expected).abs().max() / expected.abs().max() < 0.008
    assert not torch.equal(baselines[0], baselines[1])
    errors = torch.full((200, 64), -1, device="cuda", dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for step in range(200):
            q.copy_(original if step % 2 == 0 else -original)
            seed.fill_(20261117 + step)
            run()
            verify[(64,)](output, baselines[step % 2], errors, step, num_warps=4)
    graph.replay()
    torch.cuda.synchronize()
    assert errors.eq(0).all(), errors.sum().item()
    assert ws.status.eq(1).all()
    print(
        json.dumps(
            dict(
                event="replay",
                window=window,
                token_map=mapping is not None,
                launches=200,
                mismatches=errors.sum().item(),
                length=131067,
                nan_tail=True,
                all_ready=True,
            )
        ),
        flush=True,
    )
    Path(
        f"/home/bangtliu/coord/logs/phase3c-S2-opt3-replay-{window}-records.json"
    ).write_text(json.dumps(errors.cpu().tolist()) + "\n")
    # Last device test: a missing CTA prevents its arrival group from completing.
    # Full: 255 fail. Sliding: the other heads complete; the missing head fails.
    skip = 191 if window < 0 else 4 * ws.splits - 1
    flag = 0 if window < 0 else skip
    previous = ws.flags[flag, 0].item()
    ws.status.fill_(-1)
    run(skip=skip)
    torch.cuda.synchronize()
    expected_status = torch.zeros_like(ws.status)
    if window < 0:
        expected_status[191] = -1
    else:
        expected_status[:192] = 1
        expected_status[256 - 64 // ws.splits :] = -1
    assert torch.equal(ws.status, expected_status), ws.status.tolist()
    assert ws.flags[flag, 0].item() == previous
    print(
        json.dumps(
            dict(
                event="missing_arrival",
                window=window,
                skipped_cta=skip,
                failed_consumers=expected_status.eq(0).sum().item(),
                absent_consumers=expected_status.eq(-1).sum().item(),
                stale_tag=previous,
                bounded=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

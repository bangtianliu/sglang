"""Configured fast native components: AITER BF16 dense plus Nithin uTLX attention."""

import argparse
import json
from pathlib import Path

import torch
from aiter.tuned_gemm import tgemm
from bench_layer_stages import timed
from check_tail import check, native_boundary, native_norm_gate
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import build
from reference_archive import checked_load

from sglang.kernels.ops.attention.qk_norm_rope_cache import qk_norm_rope_cache
from sglang.kernels.ops.attention.tlx.paged import unified_attention


def benchmark(f, w, m, ref, context, left, rank):
    def dense(x, weight):
        return tgemm.mm(x, weight, None, otype=torch.bfloat16)

    def s1():
        p = dense(f.x, w["qkvg"])
        q, k, v, g = p.split((2048, 256, 256, 2048), -1)
        qo, ko = qk_norm_rope_cache(
            q,
            k,
            v,
            f.positions,
            m[1],
            f.cache_k,
            f.cache_v,
            f.locations,
            m[3],
            1e-5,
            5.4730064863875,
            f.k_scale,
            f.v_scale,
        )
        return qo, ko, v, g

    values = s1()
    for name, t in zip(("q", "k", "v", "gate"), values):
        check(t.reshape(8, -1), ref[name], 0.015)
    result = {"S1": timed(s1)}
    qa = ref["q"].reshape(8, 32, 64).contiguous()
    aa = ref["attn"].bfloat16().contiguous()
    k, v = f.cache_k, f.cache_v
    if left >= 0:
        valid = f.mapping >= 0
        k = torch.zeros(
            (context // 16, 16, 4, 64), device="cuda", dtype=torch.float8_e4m3fn
        )
        v = torch.zeros_like(k)
        k.view(torch.uint8).reshape(context, 4, 64)[valid] = f.cache_k.view(
            torch.uint8
        ).reshape(-1, 4, 64)[f.mapping[valid]]
        v.view(torch.uint8).reshape(context, 4, 64)[valid] = f.cache_v.view(
            torch.uint8
        ).reshape(-1, 4, 64)[f.mapping[valid]]
    out = torch.empty((8, 32, 64), device="cuda", dtype=torch.bfloat16)

    def s2():
        return unified_attention(
            q=qa,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=f.query_start,
            seqused_k=f.seq_lens,
            max_seqlen_q=8,
            max_seqlen_k=context,
            softmax_scale=0.125,
            causal=True,
            window_size=(left, 0),
            block_table=f.page_table,
            k_descale=f.k_scale,
            v_descale=f.v_scale,
            softcap=0.0,
            q_descale=None,
        )

    s2()
    check(out.reshape(8, 2048), ref["attn"], 0.008)
    result["S2"] = timed(s2)

    def s3():
        part = dense(native_norm_gate(aa, ref["gate"]), w["o_proj"])
        value = torch.zeros_like(part, dtype=torch.float32)
        for src in range(4):
            value = value + (part if src == rank else ref["partial3"][src]).float()
        return native_boundary(
            value.bfloat16(), f.stream, w["pre3"], w["alpha3"], w["beta3"]
        )

    for a, b in zip(s3(), (ref["x4"], ref["mid_stream"])):
        check(a, b, 0.02)
    result["S3"] = timed(s3)

    def s6():
        return native_boundary(
            dense(ref["x6"], w["unpooler"]),
            ref["mid_stream"],
            w["pre6"],
            w["alpha6"],
            w["beta6"],
            post=w["post6"],
        )

    for a, b in zip(s6(), (ref["out"], ref["stream"])):
        check(a, b, 0.02)
    result["S6"] = timed(s6)
    return result


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    a = p.parse_args()
    root = Path("/home/bangtliu/coord/logs")
    path = root / f"phase43-stage-sums-{a.context}-{a.left}.json"
    rows = json.loads(path.read_text())
    assert len(rows) == 4
    f, w, m = make(7, a.context, a.left, 0)
    ref = build(f, w, 7, a.context, a.left, expert_fp64=False)
    wo = checked_load("S3-inputs-7.pt")["weight"]
    for rank in range(4):
        w["o_proj"] = wo[rank]
        fast = benchmark(f, w, m, ref, a.context, a.left, rank)
        # Routing and expert backends are already the same native implementation.
        fast.update({n: rows[rank]["native"][n] for n in ("S4", "S5")})
        rows[rank]["native_fast"] = fast
        rows[rank]["native_fast_sum_us"] = sum(v["median_us"] for v in fast.values())
        rows[rank]["native_fast_backend"] = (
            "AITER tgemm BF16; SGLang uTLX paged; native router/MoE; GPU-local ordered sums"
        )
        print(json.dumps(rows[rank]), flush=True)
        path.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

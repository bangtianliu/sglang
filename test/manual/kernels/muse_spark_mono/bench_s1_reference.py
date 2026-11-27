"""Unfused SGLang QKV linear plus native norm/RoPE/cache reference on GPU0."""

import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from layer_fixture import EPS, Q_SCALE, build_layer

from sglang.kernels.ops.attention.qk_norm_rope_cache import qk_norm_rope_cache
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod


def project(f):
    # A real local Linear owns the native weight; no TP/process group is constructed.
    linear = torch.nn.Linear(8192, 4608, bias=False, device="meta")
    linear.weight = torch.nn.Parameter(f.weights["qkvg"], requires_grad=False)
    return UnquantizedLinearMethod().apply(linear, f.x, bias=None)


def reference(f, *, rope, mapping, cache_k=None, cache_v=None, projected=None):
    if projected is None:
        projected = project(f)
    q, k, v, gate = projected.split((2048, 256, 256, 2048), dim=-1)
    qo, ko = qk_norm_rope_cache(
        q,
        k,
        v,
        f.positions,
        f.cos_sin if rope else None,
        f.cache_k if cache_k is None else cache_k,
        f.cache_v if cache_v is None else cache_v,
        f.locations,
        f.mapping if mapping else None,
        EPS,
        Q_SCALE,
        f.k_scale,
        f.v_scale,
    )
    return qo, ko, v, gate


def main():
    f = build_layer()
    rows = []
    for rope, mapping in ((False, False), (True, True)):
        timing = graph_times(lambda: reference(f, rope=rope, mapping=mapping))
        row = dict(rope=rope, mapping=mapping, **timing)
        print(json.dumps(row), flush=True)
        rows.append(row)
    Path("/home/bangtliu/coord/logs/phase3-S1-unfused-reference.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

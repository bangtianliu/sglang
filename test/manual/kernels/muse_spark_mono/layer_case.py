"""Coherent layer inputs from hashed stage fixtures; one GPU, four reference ranks."""

import torch
from reference_archive import checked_load, fixture


def make(seed, context, left, rank):
    f = fixture(seed, context)
    d3 = checked_load(f"S3-inputs-{seed}.pt")
    d4 = checked_load(f"phase3d-S4-inputs-{seed}.pt")
    d5 = checked_load(f"phase3e1-inputs-{seed}.pt")
    d6 = checked_load(f"S6-inputs-{seed}.pt")
    gen = torch.Generator().manual_seed(seed)
    gamma = (torch.randn(4096, generator=gen) * 0.02).to(
        device="cuda", dtype=torch.float32 if seed == 7 else torch.bfloat16
    )
    weights = dict(
        qkvg=f.weights["qkvg"],
        o_proj=d3["weight"][rank],
        joint_router_pooler=d4["weight"],
        unpooler=d6["weight"],
        pre3=d3["pre"],
        alpha3=d3["alpha"],
        beta3=d3["beta"],
        bias=d4["bias"],
        route_norm=d4["norm"],
        gamma5=gamma,
        post6=d6["post"],
        pre6=d6["pre"],
        alpha6=d6["alpha"],
        beta6=d6["beta"],
    )
    weights.update(
        {k: v.roll(rank * 11, 0).contiguous() for k, v in d5["weights"].items()}
    )
    if left >= 0:
        first = max(0, context - 8 - left) // 16 * 16
        logical = torch.arange(first, context, device="cuda")
        source = f.page_table.flatten()[logical // 16].long() * 16 + logical % 16
        gen_gpu = torch.Generator(device="cuda").manual_seed(seed + 123)
        allocation = torch.randperm(logical.numel(), generator=gen_gpu, device="cuda")
        mapping = torch.full((context,), -1, device="cuda", dtype=torch.int64)
        mapping[source] = allocation
        for name in ("cache_k", "cache_v"):
            old = f.cache_k if name == "cache_k" else f.cache_v
            new = torch.empty((len(logical), 4, 64), device="cuda", dtype=torch.uint8)
            new[allocation] = old.view(torch.uint8).reshape(context, 4, 64)[source]
            value = new.view(torch.float8_e4m3fn).reshape(-1, 16, 4, 64)
            if name == "cache_k":
                f.cache_k = value
            else:
                f.cache_v = value
        f.mapping = mapping
    meta = (
        f.positions,
        f.cos_sin if left >= 0 else None,
        f.locations,
        f.mapping if left >= 0 else None,
        f.cache_k,
        f.cache_v,
        f.k_scale,
        f.v_scale,
        f.page_table,
        f.seq_lens,
    )
    return f, weights, meta

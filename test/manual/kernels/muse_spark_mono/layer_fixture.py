"""Seeded TP4/EP4 native-layout tensors; no model or process-group initialization."""

import math
import os

import msgspec
import torch

EPS = 1e-5
POST_EPS = 1e-8
Q_SCALE = 5.4730064863875
ROUTER_MULTIPLIER = 0.17677669529663687
ROUTE_SCALE = 1.0


class LayerFixture(msgspec.Struct):
    weights: dict[str, torch.Tensor]
    x: torch.Tensor
    stream: torch.Tensor
    positions: torch.Tensor
    locations: torch.Tensor
    mapping: torch.Tensor
    cos_sin: torch.Tensor
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    query_start: torch.Tensor


def require_gpu0():
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise RuntimeError("this manual test requires HIP_VISIBLE_DEVICES=0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("this manual test requires CUDA_VISIBLE_DEVICES=0")
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")


def build_layer(*, seed=20261117, context=8192, components=("qkvg",), rank=0):
    require_gpu0()
    assert context >= 2048 and context % 16 == 0 and 0 <= rank < 4
    generator = torch.Generator(device="cuda:0").manual_seed(seed)

    def normal(shape, scale=1.0, dtype=torch.bfloat16):
        return (
            torch.randn(shape, generator=generator, device="cuda:0", dtype=dtype)
            * scale
        )

    weights = {}
    for name, shape in {
        "qkvg": (4608, 8192),
        "o_proj": (8192, 2048),
        "router": (8192, 256),
        "pooler": (4096, 8192),
        "unpooler": (8192, 4096),
    }.items():
        if name in components:
            fan_in = shape[0] if name == "router" else shape[1]
            weights[name] = normal(
                shape,
                1 / math.sqrt(fan_in),
                torch.float32 if name == "router" else torch.bfloat16,
            )
    if "router" in components and "pooler" in components:
        router = weights["router"].T
        high = router.to(torch.bfloat16)
        low = (router - high.float()).to(torch.bfloat16)
        packed = torch.cat((weights["pooler"], high, low), dim=0).contiguous()
        weights["joint_router_pooler"] = packed
        weights["pooler"] = packed[:4096]
    if "experts" in components:
        for name, shape in {"w13": (64, 8192, 2048), "w2": (64, 4096, 2048)}.items():
            weights[name] = torch.randint(
                0, 256, shape, generator=generator, device="cuda:0", dtype=torch.uint8
            )
            weights[name + "_scale"] = torch.randint(
                118,
                123,
                (*shape[:-1], 128),
                generator=generator,
                device="cuda:0",
                dtype=torch.uint8,
            )
    if "norms" in components:
        for name, width in {
            "pre_attn": 8192,
            "pre_ffn": 8192,
            "post_ffn": 8192,
            "pre_expert": 4096,
            "post_expert": 4096,
        }.items():
            weights[name + "_gamma"] = normal((width,), 0.02, torch.float32)
            weights[name + "_effective"] = weights[name + "_gamma"] + 1
        for name in ("attn", "ffn"):
            gate = normal((8192,), 0.1, torch.float32) / 0.3
            beta = gate.sigmoid()
            weights[name + "_alpha"] = (
                ((-gate).sigmoid() * (1 + beta)).clamp_min(1e-3).sqrt()
            )
            weights[name + "_beta"] = beta
    if "router" in components:
        weights["expert_bias"] = normal((256,), 0.01, torch.float32)
        weights["local_expert_ids"] = torch.arange(
            rank * 64, (rank + 1) * 64, device="cuda:0", dtype=torch.int32
        )
    positions = torch.arange(context - 8, context, device="cuda:0", dtype=torch.int64)
    # A nonidentity page permutation catches confusing logical and physical slots.
    pages = torch.randperm(context // 16, generator=generator, device="cuda:0")
    mapping = torch.full((context,), -1, device="cuda:0", dtype=torch.int64)
    slots = torch.arange(2048, device="cuda:0")
    recent_physical = (
        pages[-128:][:, None] * 16 + torch.arange(16, device="cuda:0")[None, :]
    )
    mapping[recent_physical.flatten()] = (slots * 17) % 2048
    frequency = 500000.0 ** (
        -torch.arange(0, 64, 2, device="cuda:0", dtype=torch.float32) / 64
    )
    angles = (
        torch.arange(context, device="cuda:0", dtype=torch.float32)[:, None]
        * frequency[None, :]
    )
    # Cache slots and RoPE positions are separate coordinate systems.
    locations = pages[-1] * 16 + torch.arange(8, 16, device="cuda:0")
    cache_shape = (context // 16, 16, 4, 64)
    cache_k = normal(cache_shape, 0.5).to(torch.float8_e4m3fn)
    cache_v = normal(cache_shape, 0.5).to(torch.float8_e4m3fn)
    return LayerFixture(
        weights,
        normal((8, 8192)),
        normal((8, 8192), dtype=torch.float32),
        positions,
        locations,
        mapping,
        torch.cat((angles.cos(), angles.sin()), dim=1),
        cache_k,
        cache_v,
        torch.tensor(0.125, device="cuda:0"),
        torch.tensor(0.25, device="cuda:0"),
        pages.to(torch.int32)[None, :],
        torch.tensor([context], device="cuda:0", dtype=torch.int32),
        torch.tensor([0, 8], device="cuda:0", dtype=torch.int32),
    )

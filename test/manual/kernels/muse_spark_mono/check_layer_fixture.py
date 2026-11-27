"""Validate native weight packing, slot mapping, and seeded fixtures on GPU0."""

import json

import torch
from layer_fixture import build_layer


def main():
    for context in (8192, 32768, 131072):
        first = build_layer(seed=7, context=context)
        second = build_layer(seed=7, context=context)
        assert torch.equal(first.x, second.x)
        assert torch.equal(first.weights["qkvg"], second.weights["qkvg"])
        assert first.weights["qkvg"].shape == (4608, 8192)
        assert first.cache_k.shape == (context // 16, 16, 4, 64)
        assert first.locations.unique().numel() == 8
        mapped = first.mapping[first.locations]
        assert mapped.unique().numel() == 8 and (mapped >= 0).all()
        expected_slots = first.page_table[0, -1].long() * 16 + torch.arange(
            8, 16, device="cuda:0"
        )
        assert torch.equal(first.locations, expected_slots)
        assert first.mapping[first.mapping >= 0].unique().numel() == 2048
        print(
            json.dumps(
                {
                    "context": context,
                    "seed_reproducible": True,
                    "paged_and_swa_slots": "pass",
                }
            ),
            flush=True,
        )
        del first, second
    layer = build_layer(
        components=(
            "qkvg",
            "o_proj",
            "router",
            "pooler",
            "unpooler",
            "norms",
            "experts",
        )
    )
    for name in ("w13", "w2"):
        weight = layer.weights[name]
        scale = layer.weights[name + "_scale"]
        assert weight.dtype == scale.dtype == torch.uint8
        assert weight.is_contiguous() and scale.is_contiguous()
        assert weight.shape[-1] * 2 == scale.shape[-1] * 32 == 4096
    packed = layer.weights["joint_router_pooler"]
    assert packed.shape == (4608, 8192) and packed.is_contiguous()
    assert layer.weights["pooler"].data_ptr() == packed.data_ptr()
    high = layer.weights["router"].T.to(torch.bfloat16)
    low = (layer.weights["router"].T - high.float()).to(torch.bfloat16)
    assert torch.equal(packed[4096:4352], high)
    assert torch.equal(packed[4352:], low)
    print(
        json.dumps(
            {"all_weight_shapes": {k: list(v.shape) for k, v in layer.weights.items()}}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

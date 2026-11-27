"""Paired block bootstrap for the small residual gains targeted in phase4.10."""

import random
import statistics


def block_verdict(result, base="baseline", candidate="candidate"):
    rng = random.Random(410)
    orders = {}
    win = True
    for order in ("AB", "BA"):
        a = [
            r["us"] for r in result["raw"] if r["label"] == base and r["order"] == order
        ]
        b = [
            r["us"]
            for r in result["raw"]
            if r["label"] == candidate and r["order"] == order
        ]
        assert len(a) == len(b) == 200
        delta = [x - y for x, y in zip(a, b)]
        blocks = [delta[i : i + 2] for i in range(0, len(delta), 2)]
        distribution = sorted(
            statistics.median(
                [v for block in rng.choices(blocks, k=len(blocks)) for v in block]
            )
            for _ in range(1000)
        )
        gain = 1 - statistics.median(b) / statistics.median(a)
        orders[order] = dict(
            reduction=gain,
            paired_median_delta_us=statistics.median(delta),
            paired_block_bootstrap95_us=[distribution[25], distribution[974]],
        )
        win = win and gain > 0 and distribution[25] > 0
    return dict(
        win=bool(win),
        orders=orders,
        rule="Positive chain median gain and paired-block bootstrap95 lower delta >0 in BOTH orders; AA/BB pairs kept together. No fixed 2% screen in phase4.10.",
    )

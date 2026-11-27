"""Compulsory TP4 layer read bounds, with copy-based estimates for comparison."""

import argparse
import json
from pathlib import Path


def layer_read_budgets():
    m, h, f = 8, 8192, 4096
    budgets = {
        "S1 QKVG": dict(
            weight_bytes=4608 * h * 2,
            scale_bytes=8,
            activation_bytes=m * h * 2 + m * 64 * 4 * 2 + m * 24,
        ),
        "S3 attention tail": dict(
            weight_bytes=h * 2048 * 2 + h * 4 * 3,
            activation_bytes=m * 2048 * 4 + m * h * 6,
        ),
        "S4 joint route/pool": dict(
            weight_bytes=4608 * h * 2 + f * 4 + 256 * 4, activation_bytes=m * h * 2
        ),
        "S6 unpool/boundary": dict(
            weight_bytes=h * f * 2 + h * 4 * 4, activation_bytes=m * f * 2 + m * h * 4
        ),
    }
    for distinct in (16, 64):
        budgets[f"S5 experts {distinct} distinct local"] = dict(
            weight_bytes=distinct * (8192 * 2048 + 4096 * 2048) + f * 4,
            scale_bytes=distinct * (8192 * 128 + 4096 * 128),
            activation_bytes=m * f * 2 + 64 * 8,
        )
    for context in (8192, 32768, 131072):
        for mode, tokens in (("full", context), ("sliding", 2048)):
            budgets[f"S2 {mode} context{context}"] = dict(
                kv_bytes=tokens * 4 * 64 * 2,
                scale_bytes=8,
                activation_bytes=8 * 32 * 64 * 2 + (context // 16) * 4,
            )
    return budgets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bandwidth",
        type=Path,
        default=Path("/home/bangtliu/coord/logs/phase3.0-read-and-copy-bandwidth.json"),
    )
    args = parser.parse_args()
    measured = json.loads(args.bandwidth.read_text())
    rate = next(
        r
        for r in measured["records"]
        if r["mib"] == 1024 and r["operation"] == "torch_copy"
    )
    read = next(
        r
        for r in measured["records"]
        if r["mib"] == 1024 and r["operation"] == "gluon_read"
    )
    stages = layer_read_budgets()
    for row in stages.values():
        row["total_read_bytes"] = sum(row.values())
        row["read_bound_us"] = row["total_read_bytes"] / (read["read_GB_s"] * 1e3)
        row["copy_aggregate_bound_us"] = row["total_read_bytes"] / (
            rate["total_read_write_GB_s"] * 1e3
        )
        row["copy_read_equivalent_us"] = row["total_read_bytes"] / (
            rate["read_GB_s"] * 1e3
        )
    print(
        json.dumps(
            dict(
                bandwidth_source=str(args.bandwidth),
                read_GB_s=read["read_GB_s"],
                aggregate_copy_GB_s=rate["total_read_write_GB_s"],
                read_equivalent_copy_GB_s=rate["read_GB_s"],
                assumptions=[
                    "Unique weights/KV/activations read once; excludes writes, intermediate spills and virtual-collective traffic.",
                    "Copy aggregate is an empirical bandwidth proxy, not a measured pure-read ceiling or a rigorous hardware bound.",
                    "16 distinct local experts is illustrative;64 is the maximal distinct local hits for8x top8; inactive experts are not read.",
                    "Sliding budgets use the planned2048-token window; actual backend offset discrepancy is recorded in recon.md.",
                ],
                stages=stages,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

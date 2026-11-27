"""CPU-only dominant read budgets from recorded routes and source-derived layouts."""

import json
from pathlib import Path

LOG = Path("/home/bangtliu/coord/logs")


def main():
    active = json.loads((LOG / "phase45-active-experts.json").read_text())
    rate = json.loads((LOG / "phase45-bandwidth-1000000000.json").read_text())[
        "records"
    ][1]["results"]["persistent"]["read_TB_s"]
    own_dense = {
        "S1": 4608 * 8192 * 2,
        "S3": 8192 * 2048 * 2,
        "S4": 4608 * 8192 * 2,
        "S6": 8192 * 4096 * 2,
    }
    ts_dense = {
        "S1": 2304 * 8192 * 2,
        "S3": 8192 * 1024 * 2,
        "S4_router": 256 * 1024 * 4,
        "S4_pool": 512 * 8192 * 2,
        "S6": 1024 * 4096 * 2,
    }
    expert = 3 * 4096 * 4096 * 17 // 32
    records = []
    for row in active:
        tokens = (
            row["context"] if row["left"] < 0 else min(row["context"], row["left"] + 8)
        )
        own_kv = tokens * 4 * 64 * 2
        ts_kv = own_kv // 2
        own = [
            sum(own_dense.values()) + u * expert + own_kv for u in row["local_distinct"]
        ]
        ts = sum(ts_dense.values()) + row["global_distinct"] * (expert // 8) + ts_kv
        records.append(
            dict(
                **row,
                ours_bytes_by_rank=own,
                tokens_union=tokens,
                tokenspeed_fp8_bytes=ts,
                tokenspeed_bf16_bytes=ts + ts_kv,
                ours_rank0_measured_bandwidth_floor_us=own[0] / rate / 1e6,
                ours_max_rank_measured_bandwidth_floor_us=max(own) / rate / 1e6,
                tokenspeed_fp8_normalized_node_floor_us=ts / rate / 1e6,
                tokenspeed_fp8_MI355_peak_floor_us=ts / 8 / 1e6,
                tokenspeed_bf16_MI355_peak_floor_us=(ts + ts_kv) / 8 / 1e6,
            )
        )
    result = dict(
        ours_dense_bytes=own_dense,
        tokenspeed_dense_bytes=ts_dense,
        ours_expert_bytes=expert,
        tokenspeed_expert_shard_bytes=expert // 8,
        node_best_measured_TB_s=rate,
        MI355_theoretical_peak_TB_s=8,
        records=records,
        assumptions=[
            "Dominant unique dense/expert weights incl MXFP4 scales plus logical K/V union read once.",
            "Excludes vector gains, activations, writes, partials, spills, flag/poll and transport traffic; not hardware byte counters.",
            "Same recorded route IDs for both layouts, a controlled traffic model; no MI355 execution or claim that TokenSpeed selects these IDs.",
            "TokenSpeed FP8 and BF16 KV bracket the supported cache choices; actual benchmark KV precision unverified.",
            "Node floor is conditional on measured large-read throughput, not a rigorous hardware lower bound; MI355 floor uses advertised peak, not measured performance.",
            "Per-expert MXFP4: three 4096x4096 matrices times (1/2+1/32) bytes; TP8 slices the intermediate dimension, EP4 keeps complete local experts.",
        ],
    )
    (LOG / "phase45-design-bytes.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

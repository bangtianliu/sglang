"""Adversarial stable expert slots against an independent CPU construction."""

import json
from pathlib import Path

import torch
from layer_fixture import require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_route as route,
)


def main():
    require_gpu0()
    ids = torch.empty(64, device="cuda", dtype=torch.int32)
    sorted_ids = torch.empty(1024, device="cuda", dtype=torch.int32)
    experts = torch.empty(64, device="cuda", dtype=torch.int32)
    total = torch.empty(1, device="cuda", dtype=torch.int32)
    rows = torch.ones(8, device="cuda", dtype=torch.int32)
    status = torch.empty(768, device="cuda", dtype=torch.int32)
    failure = torch.zeros(2, device="cuda", dtype=torch.int32)
    generator = torch.Generator().manual_seed(2026111809)
    count = 0
    for rank in range(4):
        patterns = [
            torch.full((64,), rank * 64, dtype=torch.int32),
            torch.arange(64, dtype=torch.int32) + rank * 64,
            torch.full((64,), -1, dtype=torch.int32),
            torch.full((64,), 256, dtype=torch.int32),
        ]
        patterns += [
            torch.randint(0, 256, (64,), generator=generator, dtype=torch.int32)
            for _ in range(128)
        ]
        for number, raw in enumerate(patterns):
            ids.copy_(raw)
            route.slots[(64,)](
                ids,
                sorted_ids,
                experts,
                total,
                rows,
                status,
                failure,
                LOCAL_START=rank * 64,
                GRID=768,
                num_warps=4,
                enable_fp_fusion=False,
            )
            expected = []
            owner = []
            for expert in range(64):
                hits = [
                    j for j, v in enumerate(raw.tolist()) if v == rank * 64 + expert
                ]
                if hits:
                    padded = (len(hits) + 15) // 16 * 16
                    expected += hits + [64] * (padded - len(hits))
                    owner += [expert] * (padded // 16)
            n = len(expected)
            expected += [64] * (1024 - n)
            owner += [-1] * (64 - len(owner))
            for name, actual, want in [
                ("Sorted", sorted_ids.cpu(), torch.tensor(expected)),
                ("Experts", experts.cpu(), torch.tensor(owner)),
                ("NumPost", total.cpu(), torch.tensor([n])),
            ]:
                bad = (actual != want).nonzero()
                assert not len(bad), (
                    name,
                    rank,
                    number,
                    bad[0].tolist() if len(bad) else None,
                )
            assert bool((status == 1).all()), ("Status", rank, number)
            assert not bool(failure.any()), ("Failure", rank, number)
            count += 1
    rows[3] = 0
    route.slots[(64,)](
        ids,
        sorted_ids,
        experts,
        total,
        rows,
        status,
        failure,
        LOCAL_START=0,
        GRID=768,
        num_warps=4,
        enable_fp_fusion=False,
    )
    assert bool((status == 0).all()), "missing producer Status"
    assert int(failure[0]) == 4, "missing producer Failure"
    result = dict(
        exact_cases=count, changing_same_workspace=True, missing_producer=True
    )
    Path("/home/bangtliu/coord/logs/phaseT9-slot-adversarial.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

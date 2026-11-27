"""S5 exact old-core comparison and full native/FP64/changing-routing suites."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from check_experts import cases
from check_experts_complete import table_for
from check_experts_routed import single_eight
from check_experts_valu import mixed
from layer_fixture import require_gpu0
from layer_reference import exact_check
from reference_archive import checked_load

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_experts as old,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    independent_experts as new,
)

LOG = Path("/home/bangtliu/coord/logs/phase49-S5-suite")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    LOG.mkdir(exist_ok=True)
    d = checked_load(f"phase3e1-inputs-{a.seed}.pt")
    x = d["x"]
    weights = d["weights"]
    gen = torch.Generator().manual_seed(a.seed)
    gamma = (torch.randn(4096, generator=gen) * 0.02).to(
        device="cuda", dtype=torch.float32 if a.seed == 7 else torch.bfloat16
    )
    scores = torch.rand((8, 8), generator=gen).cuda()
    scores /= scores.sum(1, keepdim=True)
    patterns = cases(a.seed, "cuda")
    patterns.update(one8=single_eight("cuda"), mixed=mixed("cuda"))
    for name in ("empty", "one8", "mixed"):
        patterns[name] = torch.where(patterns[name] == 255, 256, patterns[name])
    peers = torch.zeros((4, 8, 4096), device="cuda", dtype=torch.bfloat16)
    records = []
    for name, ids in patterns.items():
        tables = table_for(ids)
        for rank in range(4):
            w = {k: v.roll(rank * 11, 0).contiguous() for k, v in weights.items()}
            table = tuple(v.reshape(4, -1)[rank] for v in tables)
            spaces = []
            for module in (old, new):
                ws = module.make_workspace("cuda")
                spaces.append(ws)
                ws.inter.fill_(float("nan"))
                ws.output.fill_(float("nan"))
                standin.prepare(ws.peers, peers, ws.state.epochs, rank=rank)
                module.s5(x, ids, scores, table, w, gamma, ws, rank=rank)
            torch.cuda.synchronize()
            valid = ((ids >= rank * 64) & (ids < (rank + 1) * 64)).flatten()
            aa, bb = spaces
            pairs = [
                ("inter", aa.inter[rank, valid], bb.inter[rank, valid]),
                ("Y", aa.output[rank, valid], bb.output[rank, valid]),
                ("scaled", aa.scaled[rank, valid], bb.scaled[rank, valid]),
                ("inv", aa.inv[rank, valid, 0], bb.inv[rank, valid, 0]),
                ("weighted", aa.weighted[rank], bb.weighted[rank]),
                ("partial", aa.partial[rank], bb.partial[rank]),
                ("reduced", aa.peers.reduced[rank], bb.peers.reduced[rank]),
                ("X6", aa.peers.reduced[rank].bfloat16(), bb.layer_out),
            ]
            for tensor, v, u in pairs:
                dtype = torch.int16 if v.dtype == torch.bfloat16 else torch.int32
                exact_check(
                    v.view(dtype),
                    u.view(dtype),
                    f"S5 {name} {tensor} bits",
                    {"_rank": rank},
                )
            records.append(dict(case=name, rank=rank, bit_exact=True))
    (LOG / f"exact-{a.seed}.json").write_text(json.dumps(records, indent=2) + "\n")
    import check_common_s5 as suite

    original_s5 = old.s5
    try:
        with (
            patch.object(old, "make_workspace", new.make_workspace),
            patch.object(old, "s5", new.s5),
            patch.object(suite, "ROOT", LOG),
            patch.object(
                sys,
                "argv",
                ["check_common_s5.py", "--seed", str(a.seed), "--replays", "200"],
            ),
        ):
            suite.main()
    finally:
        old.s5 = original_s5


if __name__ == "__main__":
    main()

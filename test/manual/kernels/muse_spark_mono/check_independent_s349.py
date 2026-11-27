"""S3 exact old-stage parity, then unchanged native/FP64/replay/missing suite."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from layer_fixture import require_gpu0
from layer_reference import exact_check
from reference_archive import checked_load

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attn_tail as old,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    independent_attn_tail as new,
)

LOG = Path("/home/bangtliu/coord/logs/phase49-S3-suite")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    LOG.mkdir(exist_ok=True)
    d = checked_load(f"S3-inputs-{a.seed}.pt")
    native = checked_load(f"S3-native-{a.seed}.pt")
    records = []
    for rank in range(4):
        for sign in (1, -1):
            spaces = []
            values = []
            for module in (old, new):
                ws = module.make_workspace("cuda")
                spaces.append(ws)
                standin.prepare(
                    ws.peers, native[sign]["partial"], ws.state.epochs, rank=rank
                )
                values.append(
                    module.attention_tail(
                        d["x"][rank] * sign,
                        d["gate"][rank],
                        d["weight"][rank],
                        d["stream"],
                        d["pre"],
                        d["alpha"],
                        d["beta"],
                        rank=rank,
                        workspace=ws,
                    )
                )
            torch.cuda.synchronize()
            for name, x, y in [
                ("projection", spaces[0].projection, spaces[1].projection),
                ("partial", spaces[0].partial[rank], spaces[1].partial[rank]),
                ("out", values[0][0], values[1][0]),
                ("stream", values[0][1], values[1][1]),
            ]:
                dtype = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
                exact_check(
                    x.view(dtype),
                    y.view(dtype),
                    f"S3 sign{sign} {name} bits",
                    {"_rank": rank},
                )
            records.append(dict(rank=rank, sign=sign, exact=True))
    (LOG / f"exact-{a.seed}.json").write_text(json.dumps(records, indent=2) + "\n")
    import check_common_s3 as suite

    with (
        patch.object(old, "make_workspace", new.make_workspace),
        patch.object(old, "attention_tail", new.attention_tail),
        patch.object(suite, "ROOT", LOG),
        patch.object(
            sys,
            "argv",
            ["check_common_s3.py", "--seed", str(a.seed), "--replays", "200"],
        ),
    ):
        suite.main()


if __name__ == "__main__":
    main()

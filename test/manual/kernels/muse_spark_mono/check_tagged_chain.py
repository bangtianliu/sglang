"""T13 adapter for the archived exact/replay/missing all-TLX chain suite."""

import argparse
import sys
from unittest.mock import patch

import check_tlx_chain as suite

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_chain,
)


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--mode", choices=("shared", "disjoint", "completed"), default="shared"
    )
    p.add_argument("--grid", type=int, default=192)
    p.add_argument("--producers", type=int, default=96)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--batch32", action="store_true")
    p.add_argument("--pair-load-only", action="store_true")
    a, remaining = p.parse_known_args()
    if a.batch32:
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
            tlx_tagged_batch_experts as tagged,
        )

    else:
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
            tlx_tagged_experts as tagged,
        )
    spaces = []
    original = tagged.make_workspace

    def make_workspace(device):
        ws = original(device)
        ws.mode, ws.grid, ws.producers = a.mode, a.grid, a.producers
        ws.diagnostics = a.diagnostics
        if a.pair_load_only:
            assert a.mode == "completed"
            ws.pair_poll = False
        if a.diagnostics:
            spaces.append(ws)
        return ws

    with (
        patch.object(tagged, "make_workspace", make_workspace),
        patch.object(tlx_chain, "tlx_experts", tagged),
        patch.object(sys, "argv", ["suite", *remaining]),
    ):
        suite.main()
    if a.diagnostics:
        import json
        from pathlib import Path

        import torch

        records = []
        for ws in spaces:
            grid = 1728 if ws.mode == "completed" else ws.grid
            counts = ws.stats[:grid].cpu()
            unique, frequencies = torch.unique(counts, return_counts=True)
            records.append(
                dict(
                    mode=ws.mode,
                    grid=grid,
                    histogram=dict(zip(unique.tolist(), frequencies.tolist())),
                    raw=counts.tolist(),
                    semantics="max retries per CTA/K512 step across this process; includes unused CTA slots",
                    failure_records=ws.record.cpu().tolist(),
                )
            )
        label = "phaseT13-diag-" + a.mode + "-g" + str(a.grid)
        Path("/home/bangtliu/coord/logs", label + ".json").write_text(
            json.dumps(records) + "\n"
        )
        print(label, "histograms", [r["histogram"] for r in records], flush=True)


if __name__ == "__main__":
    main()

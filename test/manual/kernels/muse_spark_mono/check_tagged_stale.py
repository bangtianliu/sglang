"""A completed epoch remains in the mailbox; a missing new producer must fail."""

import argparse
import json
from pathlib import Path

import torch
from check_chained47 import BASE
from layer_case import make
from layer_fixture import require_gpu0
from reference_archive import gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_peers
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_tagged_experts as tagged,
)
from sglang.srt.rocm_muse_spark import muse_moe_triton as moe


def main():
    require_gpu0()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("shared", "disjoint", "completed"), required=True
    )
    args = parser.parse_args()
    f, w, m = make(7, 8192, -1, 0)
    old = gpu(torch.load(BASE / "layer-7-8192--1.pt", weights_only=True))["rows"][0]
    ids, scores, x = old["ids"], old["scores"], old["pool"]
    table = moe.moe_align_block_size_small(ids, 64, 0, 16)
    ws = tagged.make_workspace("cuda")
    ws.mode = args.mode
    ws.grid = 384
    ws.producers = 256
    ws.diagnostics = True
    tlx_peers.prepare(ws.peers, old["partial5"], ws.state.epochs, rank=0)
    tagged.s5(x, ids, scores, table, w, w["gamma5"], ws, rank=0)
    torch.cuda.synchronize()
    assert int(ws.local_failure[0]) == 0
    torch.testing.assert_close(
        ws.layer_out.view(torch.int16),
        old["buffers"]["x6"].view(torch.int16),
        rtol=0,
        atol=0,
    )
    previous = int(ws.state.epochs[0, 0]) & 0xFFFFFFFF
    old_pairs = ws.pairs.clone()
    ws.layer_out.fill_(float("nan"))
    # No mailbox clear, epoch rewrite or launch initializer: only stand-in
    # collective payload preparation, exactly as the ordinary virtual chain.
    tlx_peers.prepare(ws.peers, old["partial5"], ws.state.epochs, rank=0)
    tagged.s5(x, ids, scores, table, w, w["gamma5"], ws, rank=0, skip=0)
    torch.cuda.synchronize()
    assert int(ws.local_failure[0]) == 5, ws.local_failure.cpu().tolist()
    assert bool(ws.layer_out.isnan().all())
    records = ws.record.cpu().reshape(-1, 8)
    late = records[records[:, 0] == previous + 1]
    assert len(late) > 0 and bool((late[:, 1] == previous).any()), late
    # Missing producer0 owns32words of the first padded block's valid rows.
    first = int(table[0][0])
    start = first * 2048
    torch.testing.assert_close(
        ws.pairs.flatten()[start : start + 32],
        old_pairs.flatten()[start : start + 32],
        rtol=0,
        atol=0,
    )
    counts = ws.stats[: 1728 if args.mode == "completed" else 384].cpu()
    unique, freq = torch.unique(counts, return_counts=True)
    result = dict(
        mode=args.mode,
        previous_tag=previous,
        expected_tag=previous + 1,
        failure=ws.local_failure.cpu().tolist(),
        stale_records=late.tolist(),
        max_retry=int(counts.max()),
        retry_histogram=dict(zip(unique.tolist(), freq.tolist())),
        record_semantics="max per CTA/K512 step across normal+missing calls; includes unused slots",
        output_nan=True,
        old_pairs_preserved=True,
        trap=False,
    )
    Path("/home/bangtliu/coord/logs/phaseT13-stale-" + args.mode + ".json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "stale_records"}),
        flush=True,
    )


if __name__ == "__main__":
    main()

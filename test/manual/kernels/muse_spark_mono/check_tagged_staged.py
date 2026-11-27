"""T13 re-entry: one-CTA producer or completed-pair consumer, exact to T11."""

import argparse
import json
from pathlib import Path

import torch
from check_chained47 import BASE
from layer_case import make
from layer_fixture import require_gpu0
from reference_archive import gpu
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_peers
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_experts as base,
)
from sglang.srt.rocm_muse_spark import muse_moe_triton as moe


def main():
    require_gpu0()
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=("produce", "consume"))
    parser.add_argument("--batch32", action="store_true")
    args = parser.parse_args()
    if args.batch32:
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
            tlx_tagged_batch_experts as tagged,
        )

    else:
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
            tlx_tagged_experts as tagged,
        )
    f, w, m = make(7, 8192, -1, 0)
    old = gpu(torch.load(BASE / "layer-7-8192--1.pt", weights_only=True))["rows"][0]
    ids, scores, x = old["ids"], old["scores"], old["pool"]
    table = moe.moe_align_block_size_small(ids, 64, 0, 16)
    ref = base.make_workspace("cuda")
    tlx_peers.prepare(ref.peers, old["partial5"], ref.state.epochs, rank=0)
    base.s5(x, ids, scores, table, w, w["gamma5"], ref, rank=0)
    torch.cuda.synchronize()
    assert int(ref.local_failure[0]) == 0
    ws = tagged.make_workspace("cuda")
    ws.inter.fill_(float("nan"))
    ws.output.fill_(float("nan"))
    if args.part == "consume":
        # Separate, explicitly completed host-prepared transport control.
        words = ref.inter[0].view(torch.int32).cpu().long() & 0xFFFFFFFF
        completed = words | (1 << 32)
        ws.pairs.copy_(completed)
        torch.cuda.synchronize()
    aa = (
        x,
        *table,
        w["w13"],
        w["w13_scale"],
        w["w2"],
        w["w2_scale"],
        ws.inter[0],
        ws.output[0],
        ws.state.status,
        ws.local_failure,
        ws.state.epochs,
        ws.pairs,
        ws.record,
        ws.stats,
    )
    opts = dict(
        GRID=1,
        MODE=args.part,
        PRODUCERS=0,
        STATUS_COUNT=1728,
        DIAGNOSTICS=True,
        PRE_POLL=True,
        SKIP=-1,
        Seed=None,
        LIMIT=4096,
        num_warps=4,
        waves_per_eu=2,
        enable_fp_fusion=False,
    )
    k = tagged.pair_phase.warmup(*aa, **opts, grid=(1,))
    resource = kernel_occupancy("T13-reentry-" + args.part, k)
    assert resource.active_ctas_per_cu >= 1
    dump(
        k,
        ("T13-batch-reentry-" if args.batch32 else "T13-reentry2-") + args.part,
        "tlx",
    )
    # Reject undefined padded poll operands before launch.
    calls = [
        l
        for l in k.asm["llir"].splitlines()
        if "asm sideeffect" in l and "global_load_b64" in l
    ]
    assert all("undef" not in l.rsplit('"(', 1)[-1] for l in calls)
    assert "global_wb" not in k.asm["amdgcn"] and "global_inv" not in k.asm["amdgcn"]
    pointers = {
        n: dict(
            start=t.data_ptr(),
            end=t.data_ptr() + t.numel() * t.element_size(),
            shape=list(t.shape),
            dtype=str(t.dtype),
        )
        for n, t in zip(tagged.pair_phase.arg_names, aa)
    }
    Path(
        (
            "/home/bangtliu/coord/logs/phaseT13-batch-reentry-"
            if args.batch32
            else "/home/bangtliu/coord/logs/phaseT13-reentry2-"
        )
        + args.part
        + "-ranges.json"
    ).write_text(json.dumps(pointers, indent=2) + "\n")
    tagged.pair_phase[(1,)](*aa, **opts)
    torch.cuda.synchronize()
    assert int(ws.local_failure[0]) == 0, ws.record.cpu().tolist()
    valid = (ids.flatten() >= 0) & (ids.flatten() < 64)
    name = "inter" if args.part == "produce" else "output"
    torch.testing.assert_close(
        getattr(ws, name)[0, valid].view(torch.int16),
        getattr(ref, name)[0, valid].view(torch.int16),
        rtol=0,
        atol=0,
    )
    if args.part == "produce":
        pairs = ws.pairs[valid]
        torch.testing.assert_close(
            (pairs >> 32), torch.ones_like(pairs), rtol=0, atol=0
        )
        torch.testing.assert_close(
            pairs & 0xFFFFFFFF,
            ref.inter[0, valid].view(torch.int32).long() & 0xFFFFFFFF,
            rtol=0,
            atol=0,
        )
    result = dict(
        part=args.part,
        grid=1,
        exact=True,
        valid_hits=int(valid.sum()),
        max_retry=int(ws.stats.max()),
        vgprs=resource.vgprs,
        lds=resource.lds_bytes,
        occupancy=resource.active_ctas_per_cu,
    )
    Path(
        (
            "/home/bangtliu/coord/logs/phaseT13-batch-reentry-"
            if args.batch32
            else "/home/bangtliu/coord/logs/phaseT13-reentry2-"
        )
        + args.part
        + ".json"
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

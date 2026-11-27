"""Bounded GPU0 missing/stale-producer diagnostics for T9 experimental consumers."""

import json
from pathlib import Path

import torch
from layer_fixture import require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_expert_overlap as expert,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_projection_overlap as row,
)


def main():
    require_gpu0()
    records = []
    for mode in ("expert", "row"):
        flags = torch.zeros(
            (4096 * 32 if mode == "expert" else 288 * 32,),
            device="cuda",
            dtype=torch.int32,
        )
        epoch = torch.ones((), device="cuda", dtype=torch.int32)
        record = torch.zeros((8,), device="cuda", dtype=torch.int32)
        failure = torch.zeros((3,), device="cuda", dtype=torch.int32)
        if mode == "expert":
            x = torch.zeros((64, 4096), device="cuda", dtype=torch.bfloat16)
            w = torch.zeros((4096, 2048), device="cuda", dtype=torch.uint8)
            scale = torch.zeros((4096, 128), device="cuda", dtype=torch.uint8)
            out = torch.full_like(x, float("nan"))
            status = torch.full((64,), -1, device="cuda", dtype=torch.int32)
            checks = torch.zeros((8192,), device="cuda", dtype=torch.int32)
            sorted_ids = torch.arange(16, device="cuda", dtype=torch.int32)
            experts = torch.zeros((1,), device="cuda", dtype=torch.int32)
            num = torch.tensor([16], device="cuda", dtype=torch.int32)

            def call():
                return expert.consume[(64,)](
                    x,
                    sorted_ids,
                    experts,
                    num,
                    w,
                    scale,
                    out,
                    status,
                    failure,
                    epoch,
                    flags,
                    record,
                    checks,
                    GRID=64,
                    WARM=False,
                    num_warps=4,
                    waves_per_eu=9,
                    enable_fp_fusion=False,
                )
        else:
            x = torch.zeros((8, 8192), device="cuda", dtype=torch.bfloat16)
            w = torch.zeros((4608, 8192), device="cuda", dtype=torch.bfloat16)
            out = torch.full((8, 4608), float("nan"), device="cuda")
            status = torch.full((288,), -1, device="cuda", dtype=torch.int32)

            def call():
                return row.project[(288,)](
                    x,
                    w,
                    out.view(torch.int32),
                    status,
                    Failure=failure,
                    N=4608,
                    K=8192,
                    STAGE=4,
                    SK=8,
                    ROUND_OUTPUT=False,
                    READY=flags,
                    EPOCH=epoch,
                    RECORD=record,
                    num_warps=2,
                    enable_fp_fusion=False,
                )

        for stale in (False, True):
            flags.fill_(int(stale))
            epoch.fill_(2 if stale else 1)
            record.zero_()
            failure.zero_()
            status.fill_(-1)
            out.fill_(float("nan"))
            call()
            torch.cuda.synchronize()
            r = record.cpu().tolist()
            assert r[0] == 1 and status.eq(0).all() and out.isnan().all(), (
                mode,
                stale,
                r,
            )
            assert int(failure[0]) == (5 if mode == "expert" else 4)
            expected, observed, polls = r[3:6] if mode == "expert" else r[2:5]
            assert (
                expected == int(epoch) and observed == int(stale) and polls == 4096
            ), (mode, stale, r)
            task = r[2] if mode == "expert" else r[1]
            records.append(
                dict(
                    mode=mode,
                    stale=stale,
                    record=r,
                    flag_base=flags.data_ptr(),
                    polled_address=flags.data_ptr() + task * 128,
                    failure=failure.cpu().tolist(),
                    terminal_nan=True,
                    all_status_blocked=True,
                    polls=4096,
                )
            )
            print("PASS", mode, "stale", stale, "record", r, flush=True)
    Path("/home/bangtliu/coord/logs/phaseT9-missing-producer.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

"""Bounded GPU0 residency proof using the actual four-wave W13 device function."""

import json
from pathlib import Path

import msgspec
import torch
from bench_hbm import graph_times
from check_experts_routed import single_eight
from layer_fixture import require_gpu0
from reference_archive import checked_load
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped, sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    StageOccupancy,
    grid_from_occupancy,
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts_async import (
    expert_tile,
)
from sglang.srt.rocm_muse_spark import muse_moe_triton as native

ROOT = Path("/home/bangtliu/coord/logs")


@g.jit
def probe(
    X,
    Sorted,
    Experts,
    W,
    Scale,
    Out,
    Storage,
    Epoch,
    Ready,
    Times,
    GRID: gl.constexpr,
    OMIT: gl.constexpr = False,
):
    pid = gl.program_id(0)
    # Each CTA writes its own output tile, so the core has no conflicting stores.
    expert_tile(
        X, Sorted, W, Scale, Out + pid * 64 * 4096, 0, gl.load(Experts), pid % 64, True
    )
    tag = gl.load(Epoch)
    start = sync.clock()
    ready = gl.full((), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    if not OMIT or pid != GRID - 1:
        ready, _, _ = scoped.counter_barrier(
            Storage, tag, 10_000_000, GRID, 12, MAX_SPINS=1_048_576
        )
    stop = sync.clock()
    gl.store(Ready + pid, ready)
    gl.store(Times + pid, stop - start)


@g.jit
def advance(Epoch):
    gl.store(Epoch, gl.load(Epoch) + 1)


def main():
    require_gpu0()
    saved = checked_load("phase3e1-inputs-7.pt")
    x, w = saved["x"], saved["weights"]
    ref = checked_load("phase3e1b-native-one8-7.pt")["inter"]
    sorted_ids, expert_ids, _ = native.moe_align_block_size_small(
        single_eight("cuda"), 64, 0, 16
    )
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    measured = json.loads((ROOT / "phase3e1f2-results.json").read_text())[0][
        "resources"
    ]
    stages = [
        StageOccupancy(r["phase"], 4, r["registers"], r["shared"], r["occupancy"])
        for r in measured
    ]
    geometry = grid_from_occupancy(cu, 4, stages)
    capacity = geometry.grid
    output = torch.empty((capacity, 64, 4096), device="cuda", dtype=torch.bfloat16)
    storage = torch.zeros(
        (capacity // 12 + 1 + capacity) * 32, device="cuda", dtype=torch.int32
    )
    epoch = torch.ones(1, device="cuda", dtype=torch.int32)
    ready = torch.zeros(capacity, device="cuda", dtype=torch.int32)
    times = torch.empty(capacity, device="cuda", dtype=torch.int64)
    args = [
        x,
        sorted_ids,
        expert_ids,
        w["w13"],
        w["w13_scale"],
        output,
        storage,
        epoch,
        ready,
        times,
    ]
    k_per_cu = geometry.ctas_per_cu
    while True:
        grid = cu * k_per_cu
        k = probe.warmup(*args, GRID=grid, num_warps=4, grid=(grid,))
        resource = kernel_occupancy("actual-W13-plus-bounded-barrier", k)
        print(
            json.dumps(
                dict(
                    event="compiled",
                    grid=grid,
                    spills=k.n_spills,
                    resources=msgspec.structs.asdict(resource),
                )
            ),
            flush=True,
        )
        if resource.active_ctas_per_cu >= k_per_cu:
            break
        assert resource.active_ctas_per_cu > 0
        k_per_cu = resource.active_ctas_per_cu
    (ROOT / "phase3e1f1-real-w13.amdgcn").write_text(k.asm["amdgcn"])

    def call():
        probe[(grid,)](*args, GRID=grid, num_warps=4)
        advance[(1,)](epoch)

    call()
    torch.cuda.synchronize()
    readiness = ready[:grid].cpu()
    # Payload verification is independent of readiness and is recorded separately.
    pid = torch.arange(grid, device="cuda")
    slot = torch.arange(0, 64, 8, device="cuda")
    column = (pid % 64)[:, None] * 64 + torch.arange(64, device="cuda")[None, :]
    got = output[pid[:, None, None], slot[None, :, None], column[:, None, :]]
    expected = ref[slot[None, :, None], column[:, None, :]]
    payload_good = got.eq(expected).all(2).all(1).cpu()
    result = dict(
        grid=grid,
        ctas_per_cu=k_per_cu,
        cu_count=cu,
        resources=msgspec.structs.asdict(resource),
        spills=k.n_spills,
        ready_count=int(readiness.sum()),
        payload_good_count=int(payload_good.sum()),
        per_cta_ready=readiness.tolist(),
        per_cta_payload_good=payload_good.tolist(),
        barrier_ticks=times[:grid].cpu().tolist(),
    )
    path = ROOT / "phase3e1f1-real-residency.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}),
        flush=True,
    )
    assert readiness.eq(1).all(), (
        "real-core bounded residency failure; per-CTA readiness saved"
    )
    assert payload_good.all(), "real-core payload mismatch; separate readiness saved"
    result["graph_total"] = graph_times(call)
    assert ready[:grid].eq(1).all()
    result["barrier_median_us"] = float(times[:grid].double().median() / 100)
    result["barrier_max_us"] = float(times[:grid].double().max() / 100)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                event="timing",
                total=result["graph_total"],
                barrier_median_us=result["barrier_median_us"],
                barrier_max_us=result["barrier_max_us"],
            )
        ),
        flush=True,
    )
    # Missing arrival and stale flags last; no trap or hardware reset.
    storage.zero_()
    storage[(grid // 12 + 1) * 32 :: 32] = 99
    epoch.fill_(1)
    ready.fill_(-1)
    probe[(grid,)](*args, GRID=grid, OMIT=True, num_warps=4)
    torch.cuda.synchronize()
    missing = ready[:grid].cpu()
    result["missing_ready_count"] = int(missing.sum())
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(dict(event="missing", ready_count=int(missing.sum()), grid=grid)),
        flush=True,
    )
    assert missing.eq(0).all()


if __name__ == "__main__":
    main()

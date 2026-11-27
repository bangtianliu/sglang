"""Vector RS+AG versus one-round BF16 exchange; GPU0, timeout -s KILL 300."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
from test_virtual_allreduce import advance, refresh, replay_time, verify

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective import (
    allreduce_plan,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective_fast import (
    batched_rs_ag,
    oneshot,
    oneshot_plan,
)

LOG = Path("/home/bangtliu/coord/logs")


def run_case(
    world,
    width,
    blocks,
    algo,
    profile,
    *,
    waves=None,
    grid_limit=None,
    log_prefix="phase2.8",
):
    grid = world * blocks
    limit = (
        grid_limit
        if grid_limit is not None
        else torch.cuda.get_device_properties(0).multi_processor_count
    )
    assert grid <= limit
    threads = 32 * waves if waves is not None else 256
    quads = width if algo == "oneshot" else 2 * width // world
    q = max(threads, triton.next_power_of_2(triton.cdiv(quads, blocks)))
    warps = waves if waves is not None else 8
    if waves is None and 4 + 4 * world * max(1, q // threads) > 128:
        warps = 16
    assert 4 + 4 * world * max(1, q // (warps * 32)) <= 128
    torch.manual_seed(0)
    base = torch.randn((world, 8, width), device="cuda", dtype=torch.bfloat16)
    # This cancellation fixture distinguishes left-associated rank order from
    # a tree/reassociated sum: ((2**24 + 1) - 2**24) + 1 is 1, not 2.
    base[:, 0, 0] = 0
    base[0, 0, 0] = 16777216
    base[1, 0, 0] = 1
    base[2, 0, 0] = -16777216
    base[3, 0, 0] = 1
    inputs = base.clone()
    reference = torch.zeros((8, width), device="cuda", dtype=torch.float32)
    for rank in range(world):
        reference = reference + base[rank].float()
    plan = (
        oneshot_plan(width, blocks, world)
        if algo == "oneshot"
        else allreduce_plan(width, blocks, world)
    )
    scratch = torch.zeros(plan.total_words, device="cuda", dtype=torch.int32)
    peers = torch.full((world,), scratch.data_ptr(), device="cuda", dtype=torch.int64)
    epoch = torch.ones(1, device="cuda", dtype=torch.int32)
    out = torch.zeros((world, 8, width), device="cuda", dtype=torch.float32)
    ready = torch.zeros((1000, grid), device="cuda", dtype=torch.int32)
    times = torch.zeros((1000, grid, 7), device="cuda", dtype=torch.int64)
    chunks = triton.cdiv(world * 8 * width, 1024)
    checks = torch.zeros((1000, chunks), device="cuda", dtype=torch.int32)
    kwargs = dict(
        WIDTH=width,
        WORLD=world,
        BPR=blocks,
        Q=q,
        PROFILE=1 if profile else 2,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    for r in plan.regions:
        kwargs[r.name.upper()] = (r.base, r.rank_stride, r.parity_stride)
    fn = oneshot if algo == "oneshot" else batched_rs_ag
    args = [
        inputs.view(torch.int32) if algo == "oneshot" else inputs,
        scratch,
        peers,
        epoch,
        out,
        ready,
        times,
    ]
    if grid_limit is not None:
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
            kernel_occupancy,
        )

        compiled = fn.warmup(*args, **kwargs, grid=(grid,))
        occupancy = kernel_occupancy("one-shot-at-expert-grid", compiled)
        assert (
            grid
            <= occupancy.active_ctas_per_cu
            * torch.cuda.get_device_properties(0).multi_processor_count
        )
    kernel = fn[(grid,)](*args, **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(out, reference.expand_as(out)), (
        "preflight",
        algo,
        world,
        width,
        blocks,
        float((out - reference).abs().max()),
    )
    asm = kernel.asm["amdgcn"]
    assert "global_load_b128" in asm and "global_store_b128" in asm
    assert "global_wb" not in asm and "global_inv" not in asm and "SCOPE_SYS" not in asm
    name = (
        f"{log_prefix}-{algo}-world{world}-width{width}-blocks{blocks}-profile{profile}"
    )
    (LOG / (name + ".amdgcn")).write_text(asm)
    scratch.zero_()
    epoch.fill_(1)
    ready.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(1000):
            refresh[(chunks,)](base, inputs, epoch, SIZE=world * 8 * width, num_warps=8)
            fn[(grid,)](*args, **kwargs)
            verify[(chunks,)](
                out,
                reference,
                epoch,
                checks,
                WIDTH=width,
                WORLD=world,
                CHUNKS=chunks,
                num_warps=8,
            )
            advance[(1,)](epoch)
    graph.replay()
    assert ready.all().item(), ("wait", algo, world, width, blocks, int(ready.sum()))
    assert checks.all().item(), (
        "numeric",
        algo,
        world,
        width,
        blocks,
        int(checks.sum()),
    )
    expected = reference * ((1 << (1000 % 7)) / 8)
    assert torch.equal(out, expected.expand_as(out))
    baseline = torch.cuda.CUDAGraph()
    with torch.cuda.graph(baseline):
        for _ in range(1000):
            advance[(1,)](epoch)
    baseline_samples = replay_time(baseline)
    measured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(measured):
        for _ in range(1000):
            fn[(grid,)](*args, **kwargs)
            advance[(1,)](epoch)
    samples = replay_time(measured)
    assert ready.all().item()
    assert torch.equal(out, expected.expand_as(out))
    data = times.double().cpu() / 100
    row = dict(
        algo=algo,
        world=world,
        width=width,
        blocks=blocks,
        warps=warps,
        q=q,
        registers=kernel.n_regs,
        shared=kernel.metadata.shared,
        profile=profile,
        checked_launches=1000,
        bit_exact=True,
        device_median_us=float(data[:, :, 6].median()),
        device_max_us=float(data[:, :, 6].max()),
        device_p99_us=float(torch.quantile(data[:, :, 6].flatten(), 0.99)),
        host_net_us=statistics.median(samples) - statistics.median(baseline_samples),
        host_samples_us=samples,
    )
    if profile:
        row["phases_us"] = {
            label: float(data[:, :, n].median())
            for n, label in enumerate(
                ("push", "wait_rs", "load_sum", "push_ag", "wait_ag", "load_ag")
            )
        }
    (LOG / (name + ".json")).write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)
    return row


def main():
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--blocks", type=int)
    p.add_argument("--algo", choices=["rs_ag", "oneshot"])
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    rows = []
    for world in (a.world,) if a.world else (4, 8):
        for width in (a.width,) if a.width else (8192, 4096):
            for blocks in (
                (a.blocks,)
                if a.blocks
                else ((8, 16, 24, 32, 48) if world == 4 else (8, 16, 24))
            ):
                for algo in (a.algo,) if a.algo else ("rs_ag", "oneshot"):
                    rows.append(run_case(world, width, blocks, algo, a.profile))
                    (
                        LOG
                        / (
                            "phase2.8-fast-profile-results.json"
                            if a.profile
                            else "phase2.8-fast-results.json"
                        )
                    ).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

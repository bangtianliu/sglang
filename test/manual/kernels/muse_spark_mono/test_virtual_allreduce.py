"""One-GPU virtual-rank parity/numerics/performance; timeout -s KILL 300."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective import (
    all_reduce,
    allreduce_plan,
)

LOG = Path("/home/bangtliu/coord/logs")


@gluon.jit
def advance(Epoch):
    gl.store(Epoch, gl.load(Epoch) + 1)


@gluon.jit
def refresh(Base, Inputs, Epoch, SIZE: gl.constexpr):
    i = gl.program_id(0) * 1024 + gl.arange(
        0, 1024, layout=gl.BlockedLayout([1], [32], [8], [0])
    )
    tag = gl.load(Epoch)
    scale = (1 << (tag % 7)).to(gl.float32) * 0.125
    gl.store(
        Inputs + i,
        gl.load(Base + i, i < SIZE, other=0).to(gl.float32) * scale,
        i < SIZE,
    )


@gluon.jit
def verify(
    Out,
    Reference,
    Epoch,
    Checks,
    WIDTH: gl.constexpr,
    WORLD: gl.constexpr,
    CHUNKS: gl.constexpr,
):
    i = gl.program_id(0) * 1024 + gl.arange(
        0, 1024, layout=gl.BlockedLayout([1], [32], [8], [0])
    )
    tag = gl.load(Epoch)
    scale = (1 << (tag % 7)).to(gl.float32) * 0.125
    got = gl.load(Out + i, i < WORLD * 8 * WIDTH, other=0)
    expected = (
        gl.load(Reference + i % (8 * WIDTH), i < WORLD * 8 * WIDTH, other=0) * scale
    )
    good = (
        gl.sum(((got == expected) | (i >= WORLD * 8 * WIDTH)).to(gl.int32), 0) == 1024
    )
    gl.store(Checks + ((tag - 1) % 1000) * CHUNKS + gl.program_id(0), good)


def replay_time(graph):
    samples = []
    for _ in range(3):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))  # ms /1000 kernels *1000 us/ms
    return samples


def run_case(width, blocks, world, profile=False):
    grid = world * blocks
    assert grid <= torch.cuda.get_device_properties(0).multi_processor_count
    torch.manual_seed(0)
    base = torch.randn((world, 8, width), device="cuda", dtype=torch.bfloat16)
    inputs = base.clone()
    reference = torch.zeros((8, width), device="cuda", dtype=torch.float32)
    for rank in range(world):
        reference = reference + base[rank].float()
    plan = allreduce_plan(width, blocks, world)
    scratch = torch.zeros(plan.total_words, device="cuda", dtype=torch.int32)
    peers = torch.full((8,), scratch.data_ptr(), device="cuda", dtype=torch.int64)
    epoch = torch.ones(1, device="cuda", dtype=torch.int32)
    out = torch.zeros((world, 8, width), device="cuda", dtype=torch.float32)
    ready = torch.zeros((1000, grid), device="cuda", dtype=torch.int32)
    chunks = triton.cdiv(world * 8 * width, 1024)
    checks = torch.zeros((1000, chunks), device="cuda", dtype=torch.int32)
    kwargs = dict(
        WIDTH=width,
        WORLD=world,
        BLOCKS_PER_RANK=blocks,
        BLOCK=triton.next_power_of_2(triton.cdiv(8 * width // world, blocks)),
        num_warps=8,
        enable_fp_fusion=False,
    )
    for name in ("partials", "partial_flags", "gathered", "gather_flags"):
        r = plan.region(name)
        kwargs[name.upper()] = (r.base, r.rank_stride, r.parity_stride)
    if profile:
        times = torch.zeros((1000, grid, 6), device="cuda", dtype=torch.int64)
        kwargs.update(PROFILE=True, TIMES=times)
    kernel = all_reduce[(grid,)](inputs, scratch, peers, epoch, out, ready, **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(out, reference.expand_as(out)), (
        "preflight",
        width,
        blocks,
        float((out - reference).abs().max()),
    )
    asm = kernel.asm["amdgcn"]
    assert "global_wb" not in asm and "global_inv" not in asm and "SCOPE_SYS" not in asm
    (
        LOG
        / f"phase{'2.8-baseline-profile' if profile else '2.5'}-world{world}-width{width}-blocks{blocks}.amdgcn"
    ).write_text(asm)
    scratch.zero_()
    epoch.fill_(1)
    ready.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(1000):
            refresh[(chunks,)](base, inputs, epoch, SIZE=world * 8 * width, num_warps=8)
            all_reduce[(grid,)](inputs, scratch, peers, epoch, out, ready, **kwargs)
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
    assert ready.all().item(), ("bounded wait failed", width, blocks, int(ready.sum()))
    assert checks.all().item(), (
        "per-launch numeric failure",
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
            all_reduce[(grid,)](inputs, scratch, peers, epoch, out, ready, **kwargs)
            advance[(1,)](epoch)
    samples = replay_time(measured)
    assert ready.all().item()
    assert torch.equal(out, expected.expand_as(out))
    row = dict(
        world=world,
        width=width,
        blocks_per_rank=blocks,
        grid=grid,
        registers=kernel.n_regs,
        shared_bytes=kernel.metadata.shared,
        scratch_bytes=plan.total_words * 4,
        checked_launches=1000,
        fixed_fp32_rank_order=list(range(world)),
        bit_exact=True,
        graph_with_epoch_us=statistics.median(samples),
        epoch_only_us=statistics.median(baseline_samples),
        allreduce_us=statistics.median(samples) - statistics.median(baseline_samples),
        samples_us=samples,
    )
    if profile:
        data = times.double().cpu() / 100
        row["phases_us"] = {
            name: dict(
                median=float(data[:, :, n].median()), maximum=float(data[:, :, n].max())
            )
            for n, name in enumerate(
                ("push", "rs_wait", "rs_load_sum", "ag_push", "ag_wait", "ag_load")
            )
        }
        (LOG / "phase2.8-baseline-phases.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)
    return row


def main():
    assert torch.cuda.device_count() == 1, "expose GPU0 only"
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--world", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--blocks", type=int)
    args = parser.parse_args()
    rows = []
    for world in (args.world,) if args.world else (4, 8):
        for width in (args.width,) if args.width else (8192, 4096):
            for blocks in (args.blocks,) if args.blocks else (8, 16, 24):
                rows.append(run_case(width, blocks, world, profile=args.profile))
                (
                    LOG
                    / (
                        "phase2.8-baseline-profile.json"
                        if args.profile
                        else "phase2.5-allreduce.json"
                    )
                ).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

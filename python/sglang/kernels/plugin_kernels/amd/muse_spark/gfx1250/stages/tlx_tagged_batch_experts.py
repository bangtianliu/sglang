"""T13 GPU0 tagged-pair W13/W2 experiment, never selected by the chain.

shared: all resident CTAs produce, then consume. disjoint: separate producer
and consumer CTA subsets. completed: two normal launches, isolating pair loads.
Fused grids fit the exact measured resident capacity; producers never wait on consumers. The old
BF16 Inter diagnostic store remains, but consumers read only tagged pairs.
"""

import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_guard as guard,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_experts as base,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_tagged_batch_core import (
    expert_tile_wide,
)


@triton.jit
def pair_phase(
    X,
    Sorted,
    Experts,
    NumPost,
    W13,
    S13,
    W2,
    S2,
    Inter,
    Out,
    Status,
    Failure,
    Epoch,
    Pairs,
    Record,
    Stats,
    GRID: tl.constexpr,
    MODE: tl.constexpr,
    PRODUCERS: tl.constexpr,
    PAIR_POLL: tl.constexpr = True,
    STATUS_COUNT: tl.constexpr = 1728,
    DIAGNOSTICS: tl.constexpr = False,
    PRE_POLL: tl.constexpr = True,
    SKIP: tl.constexpr = -1,
    Seed=None,
    LIMIT: tl.constexpr = 4096,
):
    pid = tl.program_id(0)
    tag = tl.load(Epoch).to(tl.uint32) + 1
    tasks = tl.load(NumPost) // 16 * 64
    allowed = guard.enabled(Failure, 5) & (tag != 0)
    local_ok = allowed
    if allowed:
        if (
            (MODE == "shared")
            | (MODE == "produce")
            | ((MODE == "disjoint") & (pid < PRODUCERS))
        ):
            stride = PRODUCERS if MODE == "disjoint" else GRID
            for task in range(pid, tasks, stride):
                block = task // 64
                expert = tl.load(Experts + block)
                guard.delay(Seed, task, 0)
                if task != SKIP:
                    expert_tile_wide(
                        X,
                        Sorted,
                        W13,
                        S13,
                        Inter,
                        block,
                        expert,
                        task % 64,
                        True,
                        SCALE_ASYNC=True,
                        Pairs=Pairs,
                        Tag=tag,
                        Failure=Failure,
                        Record=Record,
                        Stats=Stats,
                        PRE_POLL=PRE_POLL,
                        DIAGNOSTICS=DIAGNOSTICS,
                        LIMIT=LIMIT,
                        PAIR_POLL=PAIR_POLL,
                    )
                else:
                    local_ok = tl.full((), False, tl.int1)
        if (
            (MODE == "shared")
            | (MODE == "consume")
            | ((MODE == "disjoint") & (pid >= PRODUCERS))
        ):
            start = pid - PRODUCERS if MODE == "disjoint" else pid
            stride = GRID - PRODUCERS if MODE == "disjoint" else GRID
            for task in range(start, tasks, stride):
                block = task // 64
                expert = tl.load(Experts + block)
                guard.delay(Seed, task, 1)
                local_ok &= expert_tile_wide(
                    Inter,
                    Sorted,
                    W2,
                    S2,
                    Out,
                    block,
                    expert,
                    task % 64,
                    False,
                    SCALE_ASYNC=True,
                    Pairs=Pairs,
                    Tag=tag,
                    Failure=Failure,
                    Record=Record,
                    Stats=Stats,
                    PRE_POLL=PRE_POLL,
                    DIAGNOSTICS=DIAGNOSTICS,
                    LIMIT=LIMIT,
                    PAIR_POLL=PAIR_POLL,
                )
    ok = local_ok
    status_index = pid + tl.arange(0, 64) * GRID
    tl.store(Status + status_index, ok, status_index < STATUS_COUNT)
    guard.finish(Failure, ok, pid, 5)


class Workspace(base.Workspace):
    pairs: object = None
    record: object = None
    stats: object = None
    mode: str = "shared"
    grid: int = 192
    producers: int = 96
    diagnostics: bool = False
    pre_poll: bool = True
    poll_limit: int = 4096
    pair_poll: bool = True


def make_workspace(device):
    b = base.make_workspace(device)
    return Workspace(
        **{n: getattr(b, n) for n in b.__struct_fields__},
        pairs=torch.zeros((64, 2048), device=device, dtype=torch.int64),
        record=torch.zeros((1728, 4, 8), device=device, dtype=torch.int32),
        stats=torch.zeros((1728, 9), device=device, dtype=torch.int32),
    )


def s5(
    x,
    ids,
    scores,
    table,
    weights,
    gamma,
    ws,
    *,
    rank,
    skip=-1,
    delay_seed=None,
    compile_only=False,
    trace=None,
    read_mode=2,
    w2_sk=1,
    task_trace=None,
    failure=None,
    guard_mode=0,
    output=None,
    weight_cache="",
):
    assert trace is None and task_trace is None and read_mode == 2 and w2_sk == 1
    assert guard_mode == 0 and weight_cache == ""
    assert ws.mode in ("shared", "disjoint", "completed")
    assert ws.pair_poll or ws.mode == "completed"
    s, p = ws.state, ws.peers
    status_grid = 1728 if ws.mode == "completed" else max(64, ws.grid)
    if s.status.numel() != status_grid:
        s.status = torch.empty(status_grid, device=x.device, dtype=torch.int32)
        s.grid = status_grid
    failure = ws.local_failure if failure is None else failure
    output = ws.layer_out if output is None else output
    sorted_ids, experts, num_post = table
    aa = (
        x,
        sorted_ids,
        experts,
        num_post,
        weights["w13"],
        weights["w13_scale"],
        weights["w2"],
        weights["w2_scale"],
        ws.inter[rank],
        ws.output[rank],
        s.status,
        failure,
        s.epochs,
        ws.pairs,
        ws.record,
        ws.stats,
    )
    opts = dict(
        STATUS_COUNT=status_grid,
        PAIR_POLL=ws.pair_poll,
        DIAGNOSTICS=ws.diagnostics,
        PRE_POLL=ws.pre_poll,
        SKIP=skip,
        Seed=delay_seed,
        LIMIT=ws.poll_limit,
        num_warps=4,
        waves_per_eu=2,
        enable_fp_fusion=False,
    )
    phases = []
    if ws.mode == "completed":
        for name in ("produce", "consume"):
            phases.append(
                (
                    name,
                    pair_phase,
                    aa,
                    dict(opts, GRID=1728, MODE=name, PRODUCERS=0),
                    1728,
                )
            )
    else:
        phases.append(
            (
                ws.mode,
                pair_phase,
                aa,
                dict(opts, GRID=ws.grid, MODE=ws.mode, PRODUCERS=ws.producers),
                ws.grid,
            )
        )
    an = (ws.output, gamma, ids, ws.scaled, ws.inv, s.status, failure)
    kn = dict(RANK=rank, Seed=delay_seed, num_warps=4, enable_fp_fusion=False)
    ar = (
        ids,
        scores,
        ws.scaled,
        ws.inv,
        ws.weighted,
        ws.partial,
        p.scratch,
        p.peers,
        p.reduced,
        s.epochs,
        s.status,
        failure,
        output,
    )
    kr = dict(
        GRID=p.ctas,
        RANK=rank,
        PAYLOAD=p.payload,
        FLAGS=p.flags,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
        Y=ws.output,
        Gamma=gamma,
        FUSE_NORM=ws.fuse_norm_sum,
    )
    if not ws.fuse_norm_sum:
        phases.append(("norm", base.norm_phase, an, kn, 64))
    phases.append(("sum_AR", base.sum_reduce, ar, kr, 8))
    key = (
        rank,
        skip,
        delay_seed is not None,
        gamma.dtype,
        ws.mode,
        ws.grid,
        ws.producers,
        ws.diagnostics,
        ws.pre_poll,
        ws.poll_limit,
        ws.pair_poll,
        ws.fuse_norm_sum,
    )
    if key not in s.admitted:
        assert ws.pairs.numel() == 131072 and ws.pairs.dtype == torch.int64
        assert ws.pairs.data_ptr() % 8 == 0
        allocations = {}
        for name, value in zip(pair_phase.arg_names, aa):
            assert value.is_contiguous(), name
            base_ptr = value.untyped_storage().data_ptr()
            end_ptr = base_ptr + value.untyped_storage().nbytes()
            assert base_ptr <= value.data_ptr() < end_ptr
            assert value.data_ptr() + value.numel() * value.element_size() <= end_ptr
            allocations[name] = dict(
                start=value.data_ptr(),
                end=value.data_ptr() + value.numel() * value.element_size(),
                storage_start=base_ptr,
                storage_end=end_ptr,
                shape=list(value.shape),
                dtype=str(value.dtype),
            )
        path = (
            Path("/home/bangtliu/coord/logs")
            / f"phaseT13-ranges-{ws.mode}-g{ws.grid}-p{os.getpid()}-w{id(ws)}.json"
        )
        path.write_text(json.dumps(allocations, indent=2) + "\n")
        records = []
        kernels = []
        for name, fn, args, kwargs, grid in phases:
            k = fn.warmup(*args, **kwargs, grid=(grid,))
            calls = [
                line
                for line in k.asm["llir"].splitlines()
                if "asm sideeffect" in line and "global_load_b64" in line
            ]
            assert all("undef" not in line.rsplit('"(', 1)[-1] for line in calls), (
                "undefined packed poll operand"
            )
            assert (
                "global_wb" not in k.asm["amdgcn"]
                and "global_inv" not in k.asm["amdgcn"]
            )
            r = kernel_occupancy("T13-" + name, k)
            assert r.active_ctas_per_cu >= 1, (name, r)
            if name in ("shared", "disjoint"):
                cu = torch.cuda.get_device_properties(x.device).multi_processor_count
                assert grid <= cu * r.active_ctas_per_cu, (grid, cu, r)
                if name == "disjoint":
                    assert 0 < ws.producers < grid
            records.append(
                dict(
                    phase=name,
                    vgprs=k.n_regs,
                    spills=k.n_spills,
                    lds=k.metadata.shared,
                    occupancy=r.active_ctas_per_cu,
                    waves=k.metadata.num_warps,
                    grid=grid,
                )
            )
            kernels.append(k)
        s.admitted[key] = records
        ws.kernels[key] = kernels
    if not compile_only:
        for name, fn, args, kwargs, grid in phases:
            fn[(grid,)](*args, **kwargs)
    return ws.kernels[key][0]

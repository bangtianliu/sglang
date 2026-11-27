"""Prepared multi-rank worker. Never run outside the supervised approval wrapper."""

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from guard import HERE, STEPS, require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.step)  # before torch or any device access
    import torch
    import torch.distributed as dist

    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == STEPS[args.step] and 0 <= rank < world
    assert os.environ["HIP_VISIBLE_DEVICES"] == ",".join(map(str, range(world)))
    torch.cuda.set_device(rank)
    assert torch.cuda.device_count() == world
    assert torch.cuda.get_device_properties(rank).gcnArchName.startswith("gfx1250")
    # Existing fixture helpers use literal cuda:0. Expose each physical GPU alone
    # in a worker subprocess instead of overriding their GPU0 safety check.
    # This worker uses an explicit current device; layer child is exec'ed below.
    if args.step == "layer4":
        env = dict(
            os.environ,
            HIP_VISIBLE_DEVICES=str(rank),
            CUDA_VISIBLE_DEVICES=str(rank),
            MUSE_BRINGUP_PHYSICAL_RANK=str(rank),
        )
        os.execve(
            sys.executable,
            [
                sys.executable,
                str(HERE / "layer_worker.py"),
                "--output",
                str(args.output),
            ],
            env,
        )
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    control = dist.group.WORLD
    rccl = dist.new_group(backend="nccl", timeout=timedelta(seconds=20))
    device = torch.device("cuda", rank)
    records = []
    allocations = []

    def event(**data):
        print(json.dumps(dict(rank=rank, **data)), flush=True)
        records.append(data)

    def barrier():
        dist.monitored_barrier(group=control, timeout=timedelta(seconds=10))

    def symmetric(nbytes):
        from sglang.srt.utils.hip_vmm_utils import HipSymmetricBuffer

        buf = HipSymmetricBuffer(nbytes, device, control, uncached=True)
        allocations.append(buf)
        buf.buffers[rank].zero_()
        torch.cuda.synchronize()
        barrier()
        pointers = torch.tensor(
            [t.data_ptr() for t in buf.buffers], dtype=torch.int64, device=device
        )
        event(
            event="mapped",
            bytes=nbytes,
            handle_type=buf.handle_type,
            pointers=pointers.cpu().tolist(),
        )
        return buf.buffers[rank].view(torch.int32), pointers

    if args.step.startswith("rccl"):
        for count in (1, 32768, 1048576):
            data = torch.full((count,), rank + 1, dtype=torch.float32, device=device)
            barrier()
            dist.all_reduce(data, group=rccl)
            torch.cuda.synchronize()
            assert bool((data == world * (world + 1) // 2).all()), (rank, count)
            event(event="rccl-exact", count=count)
    elif args.step == "ping2":
        from peer_kernels import ping

        scratch, pointers = symmetric(128)
        status = torch.zeros(3, dtype=torch.int32, device=device)
        ping.warmup(scratch, pointers, status, rank, grid=(1,), num_warps=4)
        barrier()
        ping[(1,)](scratch, pointers, status, rank, num_warps=4)
        torch.cuda.synchronize()
        event(event="flag-ping", status=status.cpu().tolist())
        assert int(status[0]) == 1, (rank, status.cpu().tolist())
    else:
        from peer_kernels import oneshot

        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_peers import (
            oneshot_plan,
        )

        for width in (4096, 8192):
            plan = oneshot_plan(width, 8, world)
            regions = [plan.region(n) for n in ("payload", "flags")]
            pay, flags = [(r.base, r.rank_stride, r.parity_stride) for r in regions]
            scratch, pointers = symmetric(plan.total_words * 4)
            # Small exactly representable integers remove reduction-tree ambiguity.
            x = (
                (torch.arange(8 * width, device=device).reshape(8, width) % 13) + rank
            ).bfloat16()
            y = torch.full_like(x, float("nan"), dtype=torch.float32)
            status = torch.zeros((8, world + 1), dtype=torch.int32, device=device)
            ref = x.float()
            dist.all_reduce(ref, group=rccl)
            for tag in (1, 2, 3):
                kernel = oneshot.warmup(
                    x,
                    scratch,
                    pointers,
                    y,
                    status,
                    width,
                    world,
                    rank,
                    tag,
                    pay,
                    flags,
                    grid=(8,),
                    num_warps=4,
                )
                asm = kernel.asm["amdgcn"]
                assert "s_wait_xcnt" in asm and "SCOPE_DEV" in asm
                assert "global_wb" not in asm and "global_inv" not in asm
                stem = args.output / f"oneshot-r{rank}-w{width}-tag{tag}"
                for ext in ("ttgir", "llir", "amdgcn"):
                    stem.with_suffix("." + ext).write_text(kernel.asm[ext])
                barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                oneshot[(8,)](
                    x,
                    scratch,
                    pointers,
                    y,
                    status,
                    width,
                    world,
                    rank,
                    tag,
                    pay,
                    flags,
                    num_warps=4,
                )
                end.record()
                torch.cuda.synchronize()
                event(
                    event="oneshot",
                    width=width,
                    tag=tag,
                    status=status.cpu().tolist(),
                    microseconds=start.elapsed_time(end) * 1000,
                )
                assert bool((status[:, 0] > 0).all()), (rank, status.cpu().tolist())
                torch.testing.assert_close(y, ref, rtol=0, atol=0)
            # One RCCL comparison, not a throughput headline from three tiny trials.
            barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            data = x.float()
            start.record()
            dist.all_reduce(data, group=rccl)
            end.record()
            torch.cuda.synchronize()
            event(
                event="rccl-comparison",
                width=width,
                microseconds=start.elapsed_time(end) * 1000,
            )
    barrier()
    (args.output / f"rank-{rank}-PASS.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    # Let process exit own VMM teardown; no automatic retry/reset or next rung.
    dist.destroy_process_group(rccl)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

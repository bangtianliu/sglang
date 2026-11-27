"""Unexecuted four-real-rank layer adapter and two independent checks."""

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from guard import REPO, require


def main():
    require("layer4")  # before all device imports
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    assert os.environ["MUSE_BRINGUP_PHYSICAL_RANK"] == str(rank)
    assert os.environ["HIP_VISIBLE_DEVICES"] == str(rank)
    assert int(os.environ["WORLD_SIZE"]) == 4
    torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    control = dist.group.WORLD
    rccl = dist.new_group(backend="nccl", timeout=timedelta(seconds=20))
    sys.path.insert(0, str(REPO / "test/manual/kernels/muse_spark_mono"))
    from layer_case import make
    from layer_reference import build, compare, exact_check, named_check, routing_check
    from regular_real import Regular

    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_peers
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
        tlx_chain as chain,
    )
    from sglang.srt.utils.hip_vmm_utils import HipSymmetricBuffer

    allocations, records = [], []

    def barrier():
        dist.monitored_barrier(group=control, timeout=timedelta(seconds=10))

    def gather(tensor):
        # No GPU/peer communication: Python objects on the Gloo control group.
        values = [None] * 4
        dist.all_gather_object(values, tensor.detach().cpu(), group=control)
        return torch.stack(values).cuda()

    def bind(stage):
        p = stage.peers
        buf = HipSymmetricBuffer(
            p.scratch.numel() * 4, torch.device("cuda", 0), control, uncached=True
        )
        allocations.append(buf)
        p.scratch = buf.buffers[rank].view(torch.int32)
        p.scratch.zero_()
        p.peers = torch.tensor(
            [t.data_ptr() for t in buf.buffers], device="cuda", dtype=torch.int64
        )
        torch.cuda.synchronize()
        barrier()

    for context, left in ((8192, -1), (8192, 2047), (131072, -1), (131072, 2047)):
        seed = 7
        f, w, m = make(seed, context, left, rank)
        # Existing fixture guards require physical GPU0; do not disable them.
        # layer_case loads immutable archived tensors directly, so its CUDA:0
        # means this worker's single visible physical device instead.
        native = build(f, w, seed, context, left, expert_fp64=True)
        native["_rank"] = rank
        oracle = chain.make_workspace("cuda", context=context, left=left)
        captured = {}

        def isolated(module, api, which):
            original = getattr(module, api)

            def call(*aa, **kw):
                st = kw["workspace"] if which == 3 else aa[6]
                shape = (4, 8, 8192 if which == 3 else 4096)
                # First local pass extracts this rank's producer payload with
                # ready zero-valued stand-ins. Then repeat with ACTUAL payloads
                # exchanged over CPU Gloo. No historical/native peer payloads.
                zero = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
                tlx_peers.prepare(st.peers, zero, st.state.epochs, rank=rank)
                original(*aa, **kw)
                torch.cuda.synchronize()
                actual = gather(st.partial[rank])
                captured[which] = actual
                tlx_peers.prepare(st.peers, actual, st.state.epochs, rank=rank)
                return original(*aa, **kw)

            return call

        with (
            patch.object(
                chain.tlx_attn_tail,
                "attention_tail",
                isolated(chain.tlx_attn_tail, "attention_tail", 3),
            ),
            patch.object(chain.tlx_experts, "s5", isolated(chain.tlx_experts, "s5", 5)),
        ):
            chain.launch(f.x, f.stream, w, m, oracle, rank=rank)
            torch.cuda.synchronize()
        expected = {n: t.clone() for n, t in oracle.buffers.items()}
        ids, scores = oracle.s4.ids.clone(), oracle.s4.scores.clone()
        real = chain.make_workspace("cuda", context=context, left=left)
        bind(real.s3)
        bind(real.s5)
        barrier()
        # prepare_peers MUST NOT be called for the real workspace.
        chain.launch(f.x, f.stream, w, m, real, rank=rank)
        torch.cuda.synchronize()
        ref = {"_rank": rank}
        exact_check(real.failure[0], 0, "real chain failure", ref)
        for n, status in enumerate(chain.statuses(real), 1):
            exact_check(status, 1, f"S{n} real status", ref)
        exact_check(real.s4.ids, ids, "real routing IDs", ref)
        exact_check(
            real.s4.scores.view(torch.int32),
            scores.view(torch.int32),
            "real score bits",
            ref,
        )
        for name, value in expected.items():
            dtype = torch.int16 if value.dtype == torch.bfloat16 else torch.int32
            exact_check(
                real.buffers[name].view(dtype),
                value.view(dtype),
                name + " transport bits",
                ref,
            )
        # Native/FP64 reference uses numerical and midpoint/near-tie contracts,
        # while the isolated transport oracle above is strictly bitwise.
        # Do not apply the virtual-only teacher_reference to physical ranks.
        numerical = compare(f, real, native, w)
        regular = Regular(
            f, w, m, context, left, (captured[3], captured[5]), rank, rccl
        )
        regular()
        torch.cuda.synchronize()
        regular_numerical = {"routing": routing_check(regular, native, w)}
        for name, value in regular.outputs.items():
            regular_numerical[name] = named_check(
                value.reshape_as(native[name]),
                native[name],
                0.03,
                "regular " + name,
                native,
            )
        graphs = {}
        for name, fn in [
            ("tlx", lambda: chain.launch(f.x, f.stream, w, m, real, rank=rank)),
            ("regular", regular),
        ]:
            barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs[name] = graph
        timing = {"tlx": [], "regular": []}
        for order in (("tlx", "regular"), ("regular", "tlx")):
            for _ in range(5):
                for name in order:
                    barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graphs[name].replay()
                    end.record()
                    torch.cuda.synchronize()
                    timing[name].append(start.elapsed_time(end) * 1000)
        exact_check(real.failure[0], 0, "replay failure", ref)
        for name, value in expected.items():
            dtype = torch.int16 if value.dtype == torch.bfloat16 else torch.int32
            exact_check(
                real.buffers[name].view(dtype),
                value.view(dtype),
                name + " replay bits",
                ref,
            )
        record = dict(
            rank=rank,
            context=context,
            left=left,
            exact_transport=True,
            native_fp64=numerical,
            regular_native_fp64=regular_numerical,
            timing_us=timing,
            timing_scope="single fixture weights, supervised smoke only; not cold headline",
        )
        print(json.dumps(record), flush=True)
        records.append(record)
    barrier()
    (args.output / f"rank-{rank}-PASS.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    dist.destroy_process_group(rccl)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

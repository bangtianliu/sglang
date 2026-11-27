"""Virtual-peer pipeline with permanent mismatch records and timing perturbation."""

import datetime
import json
import unittest
from pathlib import Path

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    mailbox,
    scoped,
    sync,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.plan import (
    RegionRequest,
    make_plan,
)
from sglang.test.test_utils import CustomTestCase

LOG = Path("/home/bangtliu/coord/logs")


@gluon.jit
def advance(Counter):
    gl.store(Counter, gl.load(Counter) + 1)


@gluon.jit
def delay(tag, pid, SALT: gl.constexpr):
    seed = (tag.to(gl.uint32) * 2246822519) ^ (pid.to(gl.uint32) * 3266489917) ^ SALT
    seed = seed ^ (seed >> 13)
    count = seed & 127
    for _ in range(count):
        sync.backoff()


@gluon.jit
def mismatch(Records, tag, rank, stage, index, expected, got):
    record = Records + (rank * 3 + stage) * 6
    first = gl.atomic_cas(
        record, gl.cast(0, gl.int64), tag.to(gl.int64), sem="relaxed", scope="gpu"
    )
    if first == 0:
        gl.store(record + 1, gl.cast(rank, gl.int64))
        gl.store(record + 2, gl.cast(stage, gl.int64))
        gl.store(record + 3, gl.cast(index, gl.int64))
        gl.store(record + 4, gl.cast(expected, gl.int64))
        gl.store(record + 5, gl.cast(got, gl.int64))


@gluon.jit
def compare(Records, words, expected, i, tag, rank, stage):
    index = gl.min(gl.where(words != expected, i, 2147483647), 0)
    good = index == 2147483647
    if not good:
        got = gl.sum(gl.where(i == index, words, 0), 0)
        want = gl.sum(gl.where(i == index, expected, 0), 0)
        mismatch(Records, tag, rank, stage, index, want, got)
    return good


@gluon.jit
def pipeline(
    Scratch,
    PeerBases,
    Counter,
    Out,
    Records,
    INPUT: gl.constexpr,
    INPUT_FLAG: gl.constexpr,
    OUTPUT: gl.constexpr,
    OUTPUT_FLAG: gl.constexpr,
    INJECT: gl.constexpr = False,
    STRESS: gl.constexpr = False,
    LAUNCHES: gl.constexpr = 10000,
    WORLD: gl.constexpr = 4,
):
    pid = gl.program_id(0)
    rank = pid // 3
    tag = gl.load(Counter)
    stage = pid % 3
    if STRESS:
        stage = (stage + tag % 3) % 3
    parity = tag % 2
    dst = (rank + 1) % WORLD
    peer = gl.load(PeerBases + dst).to(gl.pointer_type(gl.int32))
    i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [8], [0]))
    mask = gl.full_like(i, True, dtype=gl.int1)
    if stage == 0:
        if STRESS:
            delay(tag, pid, 1)
        mailbox.put_words_peer(
            peer,
            INPUT[0] + dst * INPUT[1] + parity * INPUT[2],
            i,
            i + tag + rank * 1000,
            mask,
        )
        if STRESS:
            delay(tag, pid, 2)
        if not INJECT:
            mailbox.raise_flag_peer(
                peer, INPUT_FLAG[0] + dst * INPUT_FLAG[1] + parity * INPUT_FLAG[2], tag
            )
    elif stage == 1:
        if STRESS:
            delay(tag, pid, 3)
        own = gl.load(PeerBases + rank).to(gl.pointer_type(gl.int32))
        seen = mailbox.wait_flag_peer(
            own,
            INPUT_FLAG[0] + rank * INPUT_FLAG[1] + parity * INPUT_FLAG[2],
            tag,
            100_000,
        )
        if seen == tag:
            words = scoped.load_words(
                Scratch + INPUT[0] + rank * INPUT[1] + parity * INPUT[2] + i
            )
            compare(
                Records,
                words,
                i + tag + ((rank + WORLD - 1) % WORLD) * 1000,
                i,
                tag,
                rank,
                1,
            )
            mailbox.put_words_peer(
                peer,
                OUTPUT[0] + dst * OUTPUT[1] + parity * OUTPUT[2],
                i,
                words * 3 + tag,
                mask,
            )
            if STRESS:
                delay(tag, pid, 2)
            mailbox.raise_flag_peer(
                peer,
                OUTPUT_FLAG[0] + dst * OUTPUT_FLAG[1] + parity * OUTPUT_FLAG[2],
                tag,
            )
        elif not INJECT:
            mismatch(Records, tag, rank, 1, -1, tag, seen)
    else:
        if STRESS:
            delay(tag, pid, 3)
        seen = mailbox.wait_flag(
            Scratch,
            OUTPUT_FLAG[0] + rank * OUTPUT_FLAG[1] + parity * OUTPUT_FLAG[2],
            tag,
            100_000,
        )
        good = False
        if seen == tag:
            words = scoped.load_words(
                Scratch + OUTPUT[0] + rank * OUTPUT[1] + parity * OUTPUT[2] + i
            )
            expected = (i + tag + ((rank + WORLD - 2) % WORLD) * 1000) * 3 + tag
            good = compare(Records, words, expected, i, tag, rank, 2)
        elif not INJECT:
            mismatch(Records, tag, rank, 2, -1, tag, seen)
        gl.store(Out + ((tag - 1) % LAUNCHES) * WORLD + rank, good & (seen == tag))


class TestMailboxes(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
            0
        ).gcnArchName.startswith("gfx1250"):
            self.skipTest("requires gfx1250")
        assert torch.cuda.device_count() == 1, "expose GPU0 only"

    def run_pipeline(self, count, stress, world):
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        artifact = LOG / f"phase2.3b-pipeline-{stamp}"
        plan = make_plan(
            world,
            (
                RegionRequest("input", 1024),
                RegionRequest("input_flag", 1),
                RegionRequest("output", 1024),
                RegionRequest("output_flag", 1),
            ),
        )
        scratch = torch.zeros(plan.total_words, device="cuda", dtype=torch.int32)
        peers = torch.full(
            (world,), scratch.data_ptr(), device="cuda", dtype=torch.int64
        )
        counter = torch.ones(1, device="cuda", dtype=torch.int32)
        out = torch.zeros((count, world), device="cuda", dtype=torch.int32)
        records = torch.zeros((world, 3, 6), device="cuda", dtype=torch.int64)
        kwargs = {
            name.upper(): (r.base, r.rank_stride, r.parity_stride)
            for name in ("input", "input_flag", "output", "output_flag")
            for r in (plan.region(name),)
        }
        kwargs.update(STRESS=stress, LAUNCHES=count, WORLD=world, num_warps=8)
        kernel = pipeline.warmup(
            scratch, peers, counter, out, records, **kwargs, grid=(3 * world,)
        )
        asm = kernel.asm["amdgcn"]
        assert (
            "global_wb" not in asm
            and "global_inv" not in asm
            and "SCOPE_SYS" not in asm
        )
        Path(f"{artifact}.amdgcn").write_text(asm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(1000):
                pipeline[(3 * world,)](scratch, peers, counter, out, records, **kwargs)
                advance[(1,)](counter)
        for _ in range(count // 1000):
            graph.replay()
        snapshots = records.cpu().tolist()
        failures = [record for rank in snapshots for record in rank if record[0]]
        result = dict(
            launches=count,
            world=world,
            stress=stress,
            seed="tag/pid hash",
            rotating_stages=stress,
            mismatch_records=failures,
            passed_results=int(out.sum()),
            artifacts=str(artifact),
        )
        # Persist before any host assertion, including a host-reference mismatch.
        Path(f"{artifact}.json").write_text(json.dumps(result, indent=2))
        assert not failures, result
        assert out.all().item(), result
        for rank in range(world):
            offset = plan.region("output").offset(rank, count)
            actual = scratch[offset : offset + 1024].cpu()
            expected = (
                torch.arange(1024, dtype=torch.int32)
                + count
                + ((rank + world - 2) % world) * 1000
            ) * 3 + count
            if not torch.equal(actual, expected):
                index = int(torch.nonzero(actual != expected)[0])
                result["host_mismatch"] = dict(
                    launch=count,
                    rank=rank,
                    stage=2,
                    index=index,
                    expected=int(expected[index]),
                    got=int(actual[index]),
                    counter=int(counter.item()),
                )
                Path(f"{artifact}.json").write_text(json.dumps(result, indent=2))
                self.fail(str(result))
        pipeline[(3 * world,)](
            scratch, peers, counter, out, records, **kwargs, INJECT=True
        )
        assert not out[0].any().item(), out[0].tolist()
        assert not records[:, :, 0].any().item()
        print(
            f"PASS {count} launches x{world}ranks x3stages stress={stress}; stale flag rejected; artifacts={artifact}",
            flush=True,
        )

    def test_pipeline_10000_launches(self):
        for world in (4, 8):
            self.run_pipeline(10000, False, world)

    def test_timing_stress_100000_launches(self):
        for world in (4, 8):
            self.run_pipeline(100000, True, world)


if __name__ == "__main__":
    unittest.main()

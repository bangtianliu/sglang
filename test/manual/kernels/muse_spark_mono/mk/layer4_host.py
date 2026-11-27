"""MK.7b: production T11 TLX chain on 4 real GPUs, all-reduces through pinned host memory.

No GPU peer/P2P/IPC (MUSE_PEER_HOST=1: SCOPE_SYS exchange in shared /dev/shm pages).
Every rank must reproduce its archived rank row bit-exactly (same check as the
single-GPU simulated-peer runs). Times one layer and 10 back-to-back layers per graph.
Launch with torchrun --nproc-per-node=4, NCCL_P2P_DISABLE=1.
"""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

assert os.environ.get("NCCL_P2P_DISABLE") == "1", "fabric must stay disabled"
RANK = int(os.environ["LOCAL_RANK"])
os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = str(RANK)
os.environ["MUSE_PEER_HOST"] = "1"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from hostmem import HostShared  # noqa: E402

CONTEXTS = [(8192, -1), (8192, 2047), (131072, -1), (131072, 2047)]


def main():
    out_dir = Path(sys.argv[1])
    only = os.environ.get("CTX")
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))

    def barrier():
        dist.monitored_barrier(timeout=timedelta(seconds=60))

    from check_chained47 import archive_check
    from layer_case import make
    from reference_archive import gpu

    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_standin
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain

    assert tlx_standin.HOST.value
    retained = []
    records = []
    for ci, (context, left) in enumerate(CONTEXTS):
        if only and only != f"{context}:{left}":
            continue
        f, w, m = make(7, context, left, RANK)
        old = gpu(torch.load(Path("/home/bangtliu/coord/phase44-baseline")
                             / f"layer-7-{context}-{left}.pt", weights_only=True)["rows"][RANK])
        ws = tlx_chain.make_workspace("cuda", context=context, left=left)
        hosts = []
        for k, (stage, width) in enumerate(((ws.s3, 8192), (ws.s5, 4096))):
            words = tlx_standin.host_words(width, stage.peers.ctas)
            h = HostShared(f"muse_mk7b_{os.getppid()}_{ci}_{k}", words * 4, RANK, barrier)
            if RANK == 0:
                h.zero()
            stage.peers.peers.fill_(h.dptr)
            hosts.append(h)
        retained.append((f, w, m, ws, hosts, old))
        barrier()

        def launch():
            tlx_chain.launch(f.x, f.stream, w, m, ws, rank=RANK)

        # compile everywhere first (cross-rank skew would time the exchange out)
        launch()
        torch.cuda.synchronize()
        barrier()
        for it in range(3):
            ws.failure.zero_()
            barrier()
            launch()
            torch.cuda.synchronize()
            st = [int(s_.min()) for s_ in tlx_chain.statuses(ws)]
            import numpy as np
            h3 = np.frombuffer(hosts[0].map, dtype=np.int32)
            fl = h3[32 * 8192:32 * 8192 + 4 * ws.s3.peers.ctas * 32].reshape(4, ws.s3.peers.ctas, 32)[:, :8, 0]
            print(json.dumps(dict(rank=RANK, it=it, failure=ws.failure.tolist(), statuses=st,
                                  s3_epochs=ws.s3.state.epochs[:8, 0].tolist(),
                                  s3_flags=fl.tolist())), flush=True)
        row = dict(rank=RANK, context=context, left=left)
        try:
            archive_check(f, ws, old, RANK, digest=False)
            row["archive_exact"] = True
        except AssertionError as e:
            row["archive_exact"] = False
            row["archive_error"] = str(e)[:200]

        def snapshot():
            d = {n: t.clone().cpu() for n, t in ws.buffers.items()
                 if n in ("x4", "mid_stream", "x6", "out", "stream", "attn")}
            d.update(ids=ws.s4.ids.clone().cpu(), scores=ws.s4.scores.clone().cpu(),
                     pool=ws.s4.pooled.clone().cpu(), failure=ws.failure.clone().cpu())
            return d

        real = snapshot()
        p3 = ws.s3.partial.view(torch.bfloat16).reshape(4, 8, 8192)[RANK].clone().cpu()
        p5 = ws.s5.partial.view(torch.bfloat16).reshape(4, 8, 4096)[RANK].clone().cpu()
        g3, g5 = [None] * 4, [None] * 4
        dist.all_gather_object(g3, p3)
        dist.all_gather_object(g5, p5)
        torch.save(dict(real=real, partial3=torch.stack(g3), partial5=torch.stack(g5)),
                   out_dir / f"real-{context}-{left}-rank{RANK}.pt")

        def same():
            now = snapshot()
            return all(torch.equal(now[k].view(-1).view(torch.uint8), real[k].view(-1).view(torch.uint8))
                       for k in real)

        barrier()
        g10 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g10):
            for _ in range(10):
                launch()
        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            launch()  # captured last, so ws.buffers point at its outputs
        retained.append((g1, g10))
        bad = 0
        for name, g, n, reps in (("ten", g10, 10, 8), ("one", g1, 1, 20)):
            ts = []
            for _ in range(reps):
                barrier()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                g.replay()
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e) * 1000 / n)
                if n == 1:
                    bad += 0 if same() else 1
            ts.sort()
            row[f"{name}_layer_us"] = round(ts[len(ts) // 2], 1)
        row["replays_differing_from_eager"] = bad
        print(json.dumps(row), flush=True)
        records.append(row)
        barrier()
    (out_dir / f"rank-{RANK}.json").write_text(json.dumps(records, indent=2) + "\n")
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

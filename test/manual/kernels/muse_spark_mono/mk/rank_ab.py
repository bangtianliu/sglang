"""MK.41 probe: per-rank (local expert slice) A/B on GPU0, warm, archive-exact each replay.
usage: rank_ab.py --context 8192 --left -1 --variants unf,base"""
import argparse
import functools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
from bench_mono import FLAGS, with_flags  # noqa: E402
from check_chained47 import BASE, archive_check  # noqa: E402
from layer_case import make  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402
from reference_archive import checked_load, gpu  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain  # noqa: E402


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--variants", default="unf,base")
    a = p.parse_args()
    names = a.variants.split(",")
    f, w, m = make(7, a.context, a.left, 0)
    saved = gpu(torch.load(BASE / f"layer-7-{a.context}-{a.left}.pt", weights_only=True))
    basew = {n: w[n] for n in ("w13", "w13_scale", "w2", "w2_scale")}
    ow = checked_load("S3-inputs-7.pt")["weight"]
    keep = []
    for rank in range(4):
        wr = dict(w)
        wr["o_proj"] = ow[rank]
        for n, t in basew.items():
            wr[n] = t.roll(rank * 11, 0).contiguous()
        old = saved["rows"][rank]
        runs = {}
        for name in names:
            ws = with_flags(FLAGS[name], functools.partial(
                tlx_chain.make_workspace, "cuda", context=a.context, left=a.left))
            prep = functools.partial(tlx_chain.prepare_peers, ws, old["partial3"], old["partial5"],
                                     rank=rank)
            run = functools.partial(with_flags, FLAGS[name], functools.partial(
                tlx_chain.launch, f.x, f.stream, wr, m, ws, rank=rank))
            prep(); run(); torch.cuda.synchronize()
            archive_check(f, ws, old, rank, digest=False)
            prep(); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            runs[name] = (prep, g, ws)
            keep.append((ws, g, wr))
        times = {n: [] for n in names}
        for r in range(a.rounds):
            for name in (names if r % 2 == 0 else names[::-1]):
                prep, g, ws = runs[name]
                prep(); torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
                times[name].append(e0.elapsed_time(e1) * 1000)
                archive_check(f, ws, old, rank, digest=False)
        med = {k: round(sorted(v)[len(v) // 2], 2) for k, v in times.items()}
        tasks = int(runs[names[0]][2].s4.num_post[0]) // 16 * 64
        print(json.dumps(dict(context=a.context, left=a.left, rank=rank, w13_tasks=tasks,
                              median_us=med,
                              ratio={n: round(med[n] / med[names[0]], 4) for n in names[1:]})),
              flush=True)


if __name__ == "__main__":
    main()

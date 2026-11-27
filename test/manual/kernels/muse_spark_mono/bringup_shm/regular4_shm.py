"""4 real ranks, SGLang regular layer only, RCCL over host SHM (no UALoE fabric).

No peer mappings, no HipSymmetricBuffer, no TLX peer protocol. Refuses to run
unless NCCL_P2P_DISABLE=1 so RCCL cannot touch the GPU fabric.
"""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

assert os.environ.get("NCCL_P2P_DISABLE") == "1", "fabric must stay disabled"
RANK = int(os.environ["LOCAL_RANK"])
os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = str(RANK)

import faulthandler  # noqa: E402

faulthandler.dump_traceback_later(200, exit=False)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "bringup"))


def main():
    out = Path(sys.argv[1])
    torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    rccl = dist.new_group(backend="nccl", timeout=timedelta(seconds=30))
    from layer_case import make
    from regular_real import Regular

    class Local(Regular):
        def collective(self, part, index):  # compute-only control: no AR
            return part.float().bfloat16()

    def barrier():
        dist.monitored_barrier(timeout=timedelta(seconds=20))

    def timed(graph, n=20):
        ts = []
        for _ in range(n):
            barrier()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            graph.replay()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    def capture(fn):
        fn()
        torch.cuda.synchronize()
        barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g

    def log(msg):
        print(json.dumps(dict(rank=RANK, progress=msg)), flush=True)

    log("init done")
    t = torch.ones(4, device="cuda")
    dist.all_reduce(t, group=rccl)
    torch.cuda.synchronize()
    log(f"eager AR ok {t.tolist()}")
    records = []
    for width in (8192, 4096):  # the two per-layer all-reduce shapes
        x = torch.randn((8, width), device="cuda", dtype=torch.bfloat16)
        log(f"capture ar {width}")
        g = capture(lambda: dist.all_reduce(x.float(), group=rccl))
        records.append(dict(rank=RANK, what=f"ar[8,{width}]fp32", us=timed(g)))
        print(json.dumps(records[-1]), flush=True)
    for context, left in ((8192, -1), (8192, 2047), (131072, -1), (131072, 2047)):
        log(f"make {context} {left}")
        f, w, m = make(7, context, left, RANK)
        row = dict(rank=RANK, context=context, left=left)
        for name, cls in (("regular_real_ar", Regular), ("regular_no_ar", Local)):
            r = cls(f, w, m, context, left, (None, None), RANK, rccl)
            row[name + "_us"] = timed(capture(r))
            if name == "regular_real_ar":
                outs = r.outputs["out"].float()
                gathered = [None] * 4
                dist.all_gather_object(gathered, float(outs.abs().sum()))
                row["finite"] = bool(torch.isfinite(outs).all())
                row["out_abs_sum_by_rank"] = gathered
        print(json.dumps(row), flush=True)
        records.append(row)
    barrier()
    (out / f"rank-{RANK}.json").write_text(json.dumps(records, indent=2) + "\n")
    dist.destroy_process_group(rccl)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

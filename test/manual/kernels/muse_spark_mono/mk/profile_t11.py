"""GPU0: per-kernel device time of the production T11 layer (rotating 3 weight copies,
so caches stay cold-ish), via torch.profiler. Prints kernels in launch order with median us."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402
from layer_case import make  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402
from reference_archive import gpu  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain  # noqa: E402


def main():
    for kv in filter(None, os.environ.get("CHAIN_FLAGS", "").split(",")):
        k, v = kv.split("=")
        owner = tlx_chain
        if "." in k:
            from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250 import stages
            mod, k = k.split(".")
            owner = getattr(stages, mod)
        setattr(owner, k, int(v) if v not in ("0", "1") else v == "1")
    require_gpu0()
    context, left = int(os.environ.get("CTX_LEN", 8192)), int(os.environ.get("LEFT", -1))
    old = gpu(torch.load(Path("/home/bangtliu/coord/phase44-baseline") / f"layer-7-{context}-{left}.pt",
                         weights_only=True)["rows"][0])
    copies = []
    for _ in range(3):
        f, w, m = make(7, context, left, 0)
        ws = tlx_chain.make_workspace("cuda", context=context, left=left)
        copies.append((f, w, m, ws))

    def run(c):
        f, w, m, ws = c
        tlx_chain.prepare_peers(ws, old["partial3"], old["partial5"], rank=0)
        tlx_chain.launch(f.x, f.stream, w, m, ws, rank=0)

    for c in copies:
        run(c)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for r in range(6):
            for c in copies:
                run(c)
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type.name == "CUDA"]
    seq = {}
    order = []
    cnt = {}
    layer_first = None
    for e in sorted(evs, key=lambda e: e.time_range.start):
        if "_prepare" in e.name or "elementwise" in e.name:
            cnt = {}
            continue
        k = cnt.get(e.name, 0)
        cnt[e.name] = k + 1
        key = f"{e.name}#{k}"
        seq.setdefault(key, []).append(e.device_time if hasattr(e, "device_time") else e.cuda_time)
        if key not in order:
            order.append(key)
    n = 18
    tot = 0
    for name in order:
        ts = seq[name]
        per = len(ts) / n
        med = sorted(ts)[len(ts) // 2]
        tot += med * per
        print(json.dumps(dict(kernel=name[:60], per_layer=round(per, 2), med_us=round(med, 1))), flush=True)
    print(json.dumps(dict(sum_kernel_us=round(tot, 1))))


if __name__ == "__main__":
    main()

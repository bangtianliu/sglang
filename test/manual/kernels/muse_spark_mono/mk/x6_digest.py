"""GPU0 (simulated peers): one layer of the default chain on the stored fixture; prints sha256 of
the expert stage's tensors. For comparing NODES: with MX13 (scaled WMMA on FP8 x FP4) the archive
check is numeric, so it cannot show a bit-level difference between two machines (MK.133)."""
import functools
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chain10 import *  # noqa: F401,F403,E402  (setup, tlx_chain, FLAGS, with_flags, archive_check, gpu)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_experts  # noqa: E402


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def main():
    out = {}
    for flags in ("mx", "bf16"):
        if flags == "bf16":
            tlx_experts.MX13 = tlx_experts.SPLIT13 = tlx_experts.FOLD13 = False
        base, native, *rest = setup(8192, -1)
        x, stream, w, m, _ = base["chained"][1].args
        ws = with_flags(FLAGS["base"], functools.partial(tlx_chain.make_workspace, "cuda", context=8192, left=-1))
        with_flags(FLAGS["base"], functools.partial(tlx_chain.prepare_peers, ws, *native.partials, rank=0))
        with_flags(FLAGS["base"], functools.partial(tlx_chain.launch, x, stream, w, m, ws, rank=0))
        torch.cuda.synchronize()
        keep = dict(x6=ws.buffers["x6"].float().clone(), out=ws.buffers["out"].float().clone())
        if flags == "mx":
            first = keep
        else:
            d = {}
            for k in keep:
                e = (first[k] - keep[k])
                d[k] = dict(differing=int((e != 0).sum()), of=e.numel(),
                            rel_l2=float(e.norm() / keep[k].norm()),
                            max_abs_over_max=float(e.abs().max() / keep[k].abs().max()))
            out["mx_vs_bf16"] = d
        out[flags] = dict(x6=sha(ws.buffers["x6"]), inter0=sha(ws.s5.inter[0]), out=sha(ws.buffers["out"]),
                          pooled=sha(ws.s4.pooled), ids=sha(ws.s4.ids), failure=int(ws.failure[0]))
    print(json.dumps(dict(event="x6_digest", **out)), flush=True)


if __name__ == "__main__":
    main()

"""MK.7b oracle (GPU0 only): replay each rank of a real 4-GPU host-AR run with
simulated peers fed the gathered real partials; outputs must match bit for bit.

usage: layer4_oracle.py <run dir> [context:left ...]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
from layer_case import make  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain  # noqa: E402


def main():
    require_gpu0()
    run = Path(sys.argv[1])
    ctxs = sys.argv[2:] or ["8192:-1", "8192:2047", "131072:-1", "131072:2047"]
    for c in ctxs:
        context, left = map(int, c.split(":"))
        for rank in range(4):
            path = run / f"real-{context}-{left}-rank{rank}.pt"
            if not path.exists():
                continue
            d = torch.load(path, weights_only=True)
            f, w, m = make(7, context, left, rank)
            ws = tlx_chain.make_workspace("cuda", context=context, left=left)
            p3 = d["partial3"].cuda().view(torch.bfloat16).view(4, 8, 8192)
            p5 = d["partial5"].cuda().view(torch.bfloat16).view(4, 8, 4096)
            tlx_chain.prepare_peers(ws, p3, p5, rank=rank)
            tlx_chain.launch(f.x, f.stream, w, m, ws, rank=rank)
            torch.cuda.synchronize()
            sim = {n: t for n, t in ws.buffers.items()}
            sim.update(ids=ws.s4.ids, scores=ws.s4.scores, pool=ws.s4.pooled, failure=ws.failure)
            diff = {}
            for k, real in d["real"].items():
                a = sim[k].cpu().contiguous().view(-1).view(torch.uint8)
                b = real.contiguous().view(-1).view(torch.uint8)
                diff[k] = int((a != b).sum())
            print(json.dumps(dict(context=context, left=left, rank=rank,
                                  bit_exact=all(v == 0 for v in diff.values()),
                                  differing_bytes=diff)), flush=True)


if __name__ == "__main__":
    main()

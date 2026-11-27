"""Trace of fused S1 (GPU0): per tile start / projection done / counted / heads done."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402
from layer_case import make  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_qkvg  # noqa: E402


FUSED = sys.argv[1] if len(sys.argv) > 1 else True


def main():
    require_gpu0()
    f, w0, m = make(7, 8192, -1, 0)
    pos, cos, loc, mapping, kc, vc, ks, vs, pages, length = m
    copies = [w0["qkvg"].clone() for _ in range(3)]
    ws = tlx_qkvg.make_workspace("cuda")
    fail = torch.zeros(2, device="cuda", dtype=torch.int32)
    tr = torch.zeros(288 * 4, device="cuda", dtype=torch.int64)
    res = []
    for it in range(12):
        tr.zero_()
        tlx_qkvg.qkvg(f.x, copies[it % 3], pos, cos, kc, vc, loc, mapping, 1e-05, 5.4730064863875,
                      ks, vs, workspace=ws, failure=fail, fused=FUSED, trace=tr)
        torch.cuda.synchronize()
        t = tr.view(288, 4).double().cpu()
        z = t[:, 0].min() - 0
        t = (t - z) / 100.0  # 100 MHz -> us
        last = t[:, 3] > t[:, 2]
        res.append(dict(proj_done=(t[:, 0].median().item(), t[:, 0].max().item()), drain=(t[:, 1] - t[:, 0]).median().item(),
                        atomic=(t[:, 2] - t[:, 1]).median().item(),
                        heads=(t[last, 3] - t[last, 2]).median().item(), heads_max=(t[last, 3] - t[last, 2]).max().item(),
                        end=t[:, 3].max().item(), nlast=int(last.sum())))
    for r in res[3:]:
        print(json.dumps({k: (tuple(round(x, 2) for x in v) if isinstance(v, tuple) else round(v, 2)) for k, v in r.items()}))


if __name__ == "__main__":
    main()

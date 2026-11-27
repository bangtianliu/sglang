"""S1 alone (GPU0): production projection+heads (2 kernels) vs fused project_heads,
cold (3 rotating weight copies), graph replay, interleaved; outputs compared bitwise."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402
from layer_case import make  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_qkvg  # noqa: E402

VARIANTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["prod", "fused", "warm"]
FUSED = {"prod": False, "fused": True, "warm": "warm", "xchg": "xchg"}


def main():
    require_gpu0()
    keep, rows = [], []
    f, w0, m = make(7, 8192, -1, 0)
    for c in range(3):
        w = {"qkvg": w0["qkvg"].clone()}
        pos, cos, loc, mapping, kc, vc, ks, vs, pages, length = m
        row = {}
        for name in VARIANTS:
            ws = tlx_qkvg.make_workspace("cuda")
            fail = torch.zeros(2, device="cuda", dtype=torch.int32)
            kcc, vcc = kc.clone(), vc.clone()
            out = {}

            def run(ws=ws, fail=fail, kcc=kcc, vcc=vcc, out=out, name=name):
                out["r"] = tlx_qkvg.qkvg(f.x, w["qkvg"], pos, cos, kcc, vcc, loc, mapping, 1e-05,
                                         5.4730064863875, ks, vs, workspace=ws, failure=fail,
                                         fused=FUSED[name])
            run(); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            row[name] = (g, out, kcc, vcc, fail)
            keep.append((ws, fail, kcc, vcc, out, g, f, w, m))
        rows.append(row)
    ts = {n: [] for n in VARIANTS}
    for r in range(30):
        for row in rows:
            for name in (VARIANTS if r % 2 == 0 else VARIANTS[::-1]):
                g = row[name][0]
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
                ts[name].append(e0.elapsed_time(e1) * 1000)
    exact = {}
    for name in VARIANTS[1:]:
        ok = True
        for row in rows:
            a, b = row[VARIANTS[0]], row[name]
            for x, y in zip(a[1]["r"] + (a[2], a[3]), b[1]["r"] + (b[2], b[3])):
                ok &= torch.equal(x.view(torch.uint8), y.view(torch.uint8))
            ok &= int(b[4].abs().sum()) == 0
        exact[name] = ok
    med = {k: round(sorted(v)[len(v) // 2], 2) for k, v in ts.items()}
    print(json.dumps(dict(median_us=med, exact_vs_first=exact)), flush=True)


if __name__ == "__main__":
    main()

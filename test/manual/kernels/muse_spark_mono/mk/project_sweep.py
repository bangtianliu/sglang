"""Projection kernel time vs weight size (GPU0): is a 72 MiB projection latency- or
bandwidth-bound? x [8, K] @ W [N, K]^T, K = 8192 and 4096, graph of 20 launches, weights rotated
over 3 copies so no launch re-reads what the previous one read (MK.79)."""
import json
import os
import time

import torch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_projection


def main():
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    rows = []
    keep = []
    for K, sk in ((8192, 4), (4096, 4)):
        x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
        for N in (1024, 2048, 4608, 8192, 16384, 32768):
            ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(3)]
            p = torch.zeros(8 * N, device="cuda", dtype=torch.int32)
            status = torch.zeros(N // 16, device="cuda", dtype=torch.int32)
            for bn, depth, warps in ((32, 4, 2), (64, 3, 4)):
                def run(w, bn=bn, depth=depth, warps=warps):
                    tlx_projection.project[(N // bn,)](
                        x, w, p, status, N=N, K=K, STAGE=6, DEPTH=depth, BN=bn, SK=sk,
                        COPY="tdm", EARLY=True, num_warps=warps, num_stages=1,
                        enable_fp_fusion=False)
                try:
                    run(ws[0]); torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for i in range(21):
                            run(ws[i % 3])
                    best = 1e9
                    for _ in range(8):
                        torch.cuda.synchronize(); t = time.perf_counter(); g.replay()
                        torch.cuda.synchronize()
                        best = min(best, (time.perf_counter() - t) / 21 * 1e6)
                    keep.append(g)
                    mib = N * K * 2 / 2**20
                    rows.append(dict(K=K, N=N, mib=round(mib), bn=bn, us=round(best, 1),
                                     tb_s=round(mib * 2**20 / best / 1e6, 2)))
                except Exception as e:  # noqa: BLE001
                    rows.append(dict(K=K, N=N, bn=bn, error=str(e)[-120:]))
            del ws
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()

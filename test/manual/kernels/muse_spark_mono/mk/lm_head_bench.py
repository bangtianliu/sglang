"""LM head on one rank (GPU0): native torch.mm vs the chain's TLX projection kernel (MK.75).

X [8, 8192] bf16, W [50512, 8192] bf16 (one TP4 vocabulary shard, 790 MiB). The TLX kernel needs
N % BN == 0, so W is padded with zero rows to 50560. Timing: a CUDA graph of REPS launches,
replayed; µs per launch. Check: bf16 logits vs native (max abs / max rel, argmax agreement)."""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_projection  # noqa: E402

V, K, NP = 50512, 8192, 50560
REPS = 20


def timed(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS):
            fn()
    best = 1e9
    for _ in range(8):
        torch.cuda.synchronize()
        t = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / REPS * 1e6)
    return best, g


def main():
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    torch.manual_seed(0)
    w = torch.zeros(NP, K, device="cuda", dtype=torch.bfloat16)
    w[:V].normal_(0, 0.02)
    x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
    wn = w[:V]
    out_n = torch.empty(8, V, device="cuda", dtype=torch.bfloat16)

    def native():
        torch.mm(x, wn.T, out=out_n)

    t_native, keep1 = timed(native)
    rows = [dict(path="native torch.mm", us=round(t_native, 1), tb_per_s=round(V * K * 2 / t_native / 1e6, 2))]
    ref = out_n.float().clone()
    keep = [keep1]
    for bn, depth, warps in ((64, 4, 4), (64, 2, 4), (32, 4, 2), (64, 3, 4)):
        p = torch.zeros(8, NP, device="cuda", dtype=torch.int32)
        status = torch.zeros(NP // 16, device="cuda", dtype=torch.int32)

        def tlx_run(p=p, status=status, bn=bn, depth=depth, warps=warps):
            tlx_projection.project[(NP // bn,)](
                x, w, p, status, N=NP, K=K, STAGE=6, DEPTH=depth, BN=bn, SK=4, COPY="tdm",
                EARLY=True, num_warps=warps, num_stages=1, enable_fp_fusion=False)

        try:
            t, g = timed(tlx_run)
            keep.append(g)
            got = p.view(torch.float32)[:, :V]
            err = (got - ref).abs()
            rows.append(dict(path=f"tlx project BN={bn} DEPTH={depth}", us=round(t, 1),
                             tb_per_s=round(V * K * 2 / t / 1e6, 2),
                             max_abs=float(err.max()), max_rel_to_max=float(err.max() / ref.abs().max()),
                             argmax_same=bool((got.argmax(1) == ref.argmax(1)).all()),
                             differing=int((got != ref).sum()), of=int(ref.numel())))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(path=f"tlx project BN={bn} DEPTH={depth}", error=str(e)[-300:]))
    # the stage as the plugin calls it: unpadded weight, 64-column tiles + the 16-column tail
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_lm_head

    wreal = wn.contiguous()
    ws = tlx_lm_head.Workspace("cuda", V)
    assert tlx_lm_head.supported(x, wreal)
    t, g = timed(lambda: tlx_lm_head.launch(x, wreal, ws))
    keep.append(g)
    got = ws.out.float()
    err = (got - ref).abs()
    ref64 = x.double() @ wreal.double().T
    rows.append(dict(path="tlx_lm_head.launch (2 launches, fp32 out)", us=round(t, 1),
                     err_vs_fp64=float((got.double() - ref64).abs().max() / ref64.abs().max()),
                     native_err_vs_fp64=float((ref.double() - ref64).abs().max() / ref64.abs().max()),
                     max_rel_to_max=float(err.max() / ref.abs().max()),
                     argmax_same=bool((got.argmax(1) == ref.argmax(1)).all()),
                     differing=int((got != ref).sum()), tail_differing=int((got[:, V - 16:] != ref[:, V - 16:]).sum()),
                     tail_nonzero=bool((got[:, V - 16:] != 0).any())))
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()

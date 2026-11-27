"""MK.56 probe: where the two host exchanges spend their time inside the real 4-GPU chain.

layer4_ab (one variant) with stamps inside s3_boundary's and sum_reduce's row CTAs; the buffers
hold the last replayed layer. Per rank, median over the 8 rows, µs:
  pre      kernel-row entry -> exchange entry (status read, own partial load)   [S3 only]
  publish  own payload to host memory + drain + own flag
  wait     until every rank's flag for this layer is seen (includes waiting for slower ranks)
  read     the 3 peers' payloads from host memory
  post     payload read -> row exit (norm / residual)                            [S3 only]
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer4_ab  # noqa: E402
import torch  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_attn_tail, tlx_experts  # noqa: E402

s3 = torch.zeros(64, device="cuda", dtype=torch.int64)
s5 = torch.zeros(64, device="cuda", dtype=torch.int64)
st = torch.zeros(64, device="cuda", dtype=torch.int64)
tlx_attn_tail.S3_XTRACE = s3
tlx_experts.S5_XTRACE = s5
tlx_experts.SUM_TRACE = st
layer4_ab.main()
torch.cuda.synchronize()
med = lambda t: round(float(t.median()) / 100, 2)  # noqa: E731
out = dict(rank=int(os.environ["LOCAL_RANK"]))
for name, buf in (("s3_boundary", s3), ("sum_reduce", s5)):
    v = buf.view(8, 8).double().cpu()
    d = dict(publish=med(v[:, 1] - v[:, 0]), wait=med(v[:, 2] - v[:, 1]), read=med(v[:, 3] - v[:, 2]),
             polls=int(v[:, 4].median()))
    if name == "s3_boundary":
        d.update(pre=med(v[:, 0] - v[:, 5]), post=med(v[:, 6] - v[:, 3]),
                 row_total=med(v[:, 6] - v[:, 5]))
    out[name] = d
t = st.view(8, 8).double().cpu()
out["sum_reduce"].update(pre_compute=med(t[:, 3] - t[:, 0]), exchange_total=med(t[:, 4] - t[:, 3]),
                         row_total=med(t[:, 5] - t[:, 0]))
print(json.dumps(out), flush=True)

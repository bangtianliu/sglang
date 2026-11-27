"""MK.41 probe: layer4_ab with expert-phase stamps (last replayed layer of each kind).
Unfused: W13 first start -> W2 last end (PHASE_TRACE). Fused: kernel span (FUSED_TRACE)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer4_ab  # noqa: E402  (sets the visible device for this rank)
import torch  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_experts  # noqa: E402

pt = (torch.zeros(1728 * 2, device="cuda", dtype=torch.int64),
      torch.zeros(1728 * 2, device="cuda", dtype=torch.int64))
ft = torch.zeros(576 * 4, device="cuda", dtype=torch.int64)
tlx_experts.PHASE_TRACE = pt
tlx_experts.FUSED_TRACE = ft
st = torch.zeros(64, device="cuda", dtype=torch.int64)
tlx_experts.SUM_TRACE = st
layer4_ab.main()
torch.cuda.synchronize()
FUSED_LAST = os.environ["VARIANTS"].split(",")[-1] == "t11" and len(os.environ["VARIANTS"].split(",")) == 1
rank = int(os.environ["LOCAL_RANK"])
sv = st.view(8, 8).double().cpu()
if FUSED_LAST:
    v = ft.view(576, 4).double().cpu()
    e0, e1 = float(v[:, 0].min()), float(v[:, 3].max())
else:
    a = pt[0].view(1728, 2).double().cpu(); b = pt[1].view(1728, 2).double().cpu()
    e0, e1 = float(a[a[:, 1] > 0][:, 0].min()), float(b[b[:, 1] > 0][:, 1].max())
s_start = float(sv[:, 0].min())
print(json.dumps(dict(rank=rank, kind="fused" if FUSED_LAST else "unfused",
                      expert_span=round((e1 - e0) / 100, 2),
                      expert_end_to_sum_start=round((s_start - e1) / 100, 2),
                      sum_span=round(float(sv[:, 5].max() - sv[:, 0].min()) / 100, 2))), flush=True)

"""MK.41 probe: layer4_ab under torch.profiler; per-kernel mean device time on each rank
(all launches of the run: eager, singles and the 10-layer graphs)."""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer4_ab  # noqa: E402
import torch  # noqa: E402

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    layer4_ab.main()
    torch.cuda.synchronize()
tot, cnt = defaultdict(float), defaultdict(int)
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        tot[e.name] += e.device_time
        cnt[e.name] += 1
rows = sorted(((round(tot[n] / cnt[n], 2), cnt[n], n[:40]) for n in tot), reverse=True)
print(json.dumps(dict(rank=int(os.environ["LOCAL_RANK"]), variants=os.environ["VARIANTS"],
                      kernels=[r for r in rows if r[1] >= 20][:20])), flush=True)

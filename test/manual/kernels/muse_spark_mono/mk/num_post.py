"""Per-rank expert task counts (blocks of 16 slots x 64 tiles) for the archived cases (GPU0).
Routing ids are global, so rank 0's ids give every rank's local experts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from bench_chained47 import setup  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain  # noqa: E402

require_gpu0()
for context, left in [(8192, -1), (8192, 2047), (131072, -1), (131072, 2047)]:
    base, native, *rest = setup(context, left)
    x, stream, w, m, _ = base["chained"][1].args
    ws = tlx_chain.make_workspace("cuda", context=context, left=left)
    tlx_chain.prepare_peers(ws, *native.partials, rank=0)
    tlx_chain.launch(x, stream, w, m, ws, rank=0)
    torch.cuda.synchronize()
    ids = ws.s4.ids.view(-1).cpu()
    row = dict(context=context, left=left, failure=ws.failure.tolist())
    for rank in range(4):
        local = ids[(ids >= rank * 64) & (ids < rank * 64 + 64)] - rank * 64
        counts = torch.bincount(local, minlength=64)
        blocks = int(((counts + 15) // 16).sum())
        row[f"r{rank}"] = dict(routes=int(local.numel()), experts=int((counts > 0).sum()),
                              tasks=blocks * 64)
    print(json.dumps(row), flush=True)

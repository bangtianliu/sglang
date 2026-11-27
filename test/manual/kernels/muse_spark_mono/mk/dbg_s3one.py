import sys, functools
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import torch
from bench_chained47 import setup
from bench_s3n import split_tail
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain
base, native, *rest = setup(8192, -1)
x, stream, w, m, _ = base["chained"][1].args
ws = tlx_chain.make_workspace("cuda", context=8192, left=-1)
sums = torch.zeros(256, device="cuda"); rowcnt = torch.zeros(256, device="cuda", dtype=torch.int32)
shim = split_tail(sums, 8, rowcnt)
tlx_chain.prepare_peers(ws, *native.partials, rank=0)
old = tlx_chain.tlx_attn_tail; tlx_chain.tlx_attn_tail = shim
tlx_chain.launch(x, stream, w, m, ws, rank=0)
tlx_chain.tlx_attn_tail = old
torch.cuda.synchronize()
print("failure", ws.failure.tolist())
print("rowcnt", rowcnt.view(8, 32)[:, 0].tolist())
print("status[0:64]", ws.s3.state.status[:64].tolist())
print("sums row0", sums[:32].tolist())
print("epochs[0:3]", ws.s3.state.epochs[:3, 0].tolist())
print("x4 finite", bool(torch.isfinite(ws.buffers["x4"].float()).all()), float(ws.buffers["x4"].float().abs().max()))
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_s3split
for dc in tlx_s3split.slice_fused.device_caches.values():
    for k in dc[0].values():
        asm = k.asm["amdgcn"].splitlines()
        hits = [i for i, l in enumerate(asm) if "atomic" in l or "s_wait_storecnt" in l]
        print("ATOMIC_LINES", len(hits))
        for i in hits[:12]:
            print("ASM", asm[i].strip())
        j = [i for i, l in enumerate(asm) if "global_atomic_add_u32" in l]
        if j:
            [print("CTX", x.strip()) for x in asm[j[0]-14:j[0]+6]]

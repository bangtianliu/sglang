import torch, triton, triton.language as tl
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_s3split import _signal_dev
@triton.jit
def k(C, O):
    pid = tl.program_id(0)
    lanes = tl.arange(0, 128)
    sig = _signal_dev(C + (pid % 8) * 32 + lanes * 0)
    tl.store(O + pid * 128 + lanes, sig)
c = torch.zeros(256, device="cuda", dtype=torch.int32); o = torch.zeros(64 * 128, device="cuda", dtype=torch.int32)
k[(64,)](c, o, num_warps=4); torch.cuda.synchronize()
print("counters", c.view(8, 32)[:, 0].tolist(), "sig unique", o.unique().tolist())

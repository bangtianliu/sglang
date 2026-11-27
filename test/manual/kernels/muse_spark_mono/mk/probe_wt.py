import torch, triton, triton.language as tl
@triton.jit
def k(X, Y):
    i = tl.arange(0, 256)
    tl.store(Y + i, tl.load(X + i), cache_modifier=".wt")
@triton.jit
def k2(X, Y):
    i = tl.arange(0, 256)
    tl.store(Y + i, tl.load(X + i))
x = torch.zeros(256, device="cuda", dtype=torch.bfloat16)
for f in (k, k2):
    c = f.warmup(x, x, grid=(1,))
    print(f.__name__, [l.strip() for l in c.asm["amdgcn"].splitlines() if "store" in l and "global" in l or "buffer_store" in l])

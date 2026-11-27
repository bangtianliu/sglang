"""Does tl.dot_scaled (e4m3 A x e2m1 B, e8m0 scales) lower to gfx1250 scaled WMMA? + numerics."""
import torch
import triton
import triton.language as tl


@triton.jit
def k(A, AS, B, BS, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M); n = tl.arange(0, N); kk = tl.arange(0, K); kb = tl.arange(0, K // 2)
    ks = tl.arange(0, K // 32)
    a = tl.load(A + m[:, None] * K + kk[None, :])
    asc = tl.load(AS + m[:, None] * (K // 32) + ks[None, :])
    b = tl.load(B + n[:, None] * (K // 2) + kb[None, :])          # [N, K/2] packed along K
    bsc = tl.load(BS + n[:, None] * (K // 32) + ks[None, :])
    acc = tl.dot_scaled(a, asc, "e4m3", b.T, bsc, "e2m1", tl.zeros((M, N), tl.float32))
    tl.store(C + m[:, None] * N + n[None, :], acc)


def e2m1_table():
    return torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def main():
    M, N, K = 16, 64, 512
    torch.manual_seed(0)
    a = (torch.randn(M, K) * 2).to(torch.float8_e4m3fn)
    asc = torch.randint(120, 130, (M, K // 32), dtype=torch.uint8)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    bsc = torch.randint(120, 130, (N, K // 32), dtype=torch.uint8)
    t = e2m1_table()
    bv = torch.stack([t[(b & 15).long()], t[(b >> 4).long()]], -1).reshape(N, K)
    af = a.float() * torch.pow(2.0, asc.float() - 127).repeat_interleave(32, 1)
    bf = bv * torch.pow(2.0, bsc.float() - 127).repeat_interleave(32, 1)
    ref = af.double() @ bf.double().T
    c = torch.empty(M, N, device="cuda")
    h = k[(1,)](a.cuda().view(torch.uint8).view(torch.float8_e4m3fn), asc.cuda(), b.cuda(), bsc.cuda(), c,
                M, N, K, num_warps=4)
    err = (c.cpu().double() - ref).abs().max().item() / ref.abs().max().item()
    asm = h.asm["amdgcn"]
    print("rel err", err)
    print({x: asm.count(x) for x in ("v_wmma_scale", "v_wmma_f32_16x16x128_f8f6f4", "v_wmma_scale_f32",
                                       "v_cvt_scale", "v_wmma_f32_16x16x32_bf16", "v_wmma")})
    print([l.strip() for l in asm.splitlines() if "wmma" in l][:4])


main()

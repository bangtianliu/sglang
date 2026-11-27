"""GPU0: how many keys does SGLang's paged attention attend for window_size=(W, 0)?
Runs the tlx paged `unified_attention` (the op behind the tlx backend) at W = 2046 (what the
Muse Spark model + backend pass today) and W = 2047, and compares each with a torch FP64
reference that attends the last N keys (own position included) for N = 2046..2049."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from layer_case import make  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402
from layer_reference import build  # noqa: E402

from sglang.kernels.ops.attention.tlx.paged import unified_attention  # noqa: E402


def main():
    require_gpu0()
    context = 8192
    f, w, m = make(7, context, 2047, 0)
    ref = build(f, w, 7, context, 2047, expert_fp64=False)
    q = ref["q"].reshape(8, 32, 64).contiguous()
    valid = f.mapping >= 0
    k = torch.zeros((context // 16, 16, 4, 64), device="cuda", dtype=torch.float8_e4m3fn)
    v = torch.zeros_like(k)
    for dst, src in ((k, f.cache_k), (v, f.cache_v)):
        dst.view(torch.uint8).reshape(context, 4, 64)[valid] = src.view(torch.uint8).reshape(
            -1, 4, 64)[f.mapping[valid]]
    # the 8 new tokens' K/V must be in the cache (the reference's own k, v rows)
    length = int(f.seq_lens[0])
    table = f.page_table.flatten().long()
    logical = torch.arange(context, device="cuda")
    phys = table[logical // 16] * 16 + logical % 16
    kq = k.view(torch.uint8).reshape(context, 4, 64)
    vq = v.view(torch.uint8).reshape(context, 4, 64)
    ks, vs = float(f.k_scale.flatten()[0]), float(f.v_scale.flatten()[0])
    kf = kq.view(torch.float8_e4m3fn).double() * ks   # [slot, 4, 64]
    vf = vq.view(torch.float8_e4m3fn).double() * vs
    kl, vl = kf[phys], vf[phys]                         # logical order
    have_new = bool((kl[length - 8:length].abs().sum() > 0))

    def reference(n_keys):
        out = torch.zeros((8, 32, 64), dtype=torch.float64, device="cuda")
        for t in range(8):
            p = length - 8 + t
            lo = max(0, p - n_keys + 1)
            kk, vv = kl[lo:p + 1], vl[lo:p + 1]         # [n, 4, 64]
            for h in range(32):
                s = (kk[:, h // 8] @ q[t, h].double()) * 0.125
                out[t, h] = torch.softmax(s, 0) @ vv[:, h // 8]
        return out

    refs = {n: reference(n) for n in (2046, 2047, 2048, 2049)}
    rows = {}
    for W in (2046, 2047):
        out = torch.empty((8, 32, 64), device="cuda", dtype=torch.bfloat16)
        unified_attention(q=q, k=k, v=v, out=out, cu_seqlens_q=f.query_start, seqused_k=f.seq_lens,
                          max_seqlen_q=8, max_seqlen_k=context, softmax_scale=0.125, causal=True,
                          window_size=(W, 0), block_table=f.page_table, k_descale=f.k_scale,
                          v_descale=f.v_scale, softcap=0.0, q_descale=None)
        torch.cuda.synchronize()
        rows[W] = {n: float((out.double() - r).abs().max() / r.abs().max()) for n, r in refs.items()}
    sep = float((refs[2047] - refs[2048]).abs().max() / refs[2048].abs().max())
    print(json.dumps(dict(new_tokens_in_cache=have_new, length=length,
                          ref_2047_vs_2048_rel=sep,
                          rel_err={f"window_size=({W},0)": {f"{n}_keys": round(e, 6) for n, e in d.items()}
                                   for W, d in rows.items()})), flush=True)


if __name__ == "__main__":
    main()

"""S5 numerics vs FP64 (GPU0, rank 0, real chain inputs): regular SGLang MoE Triton path,
TLX bf16-activation path (base) and TLX MXFP8-activation W13 (MX13).

Reference: same x4 rows, routing table and MXFP4 weights (dequantized exactly), FP64 math.
W13 out (inter, silu(g)*u per slot) and W2 out (y per slot, from the FP64 inter)."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
from bench_chained47 import setup  # noqa: E402
from bench_mono import with_flags  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain  # noqa: E402
from sglang.srt.rocm_muse_spark import muse_moe_triton as native  # noqa: E402

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                    dtype=torch.float64)


def deq(w, s, e, N):
    """expert e of packed MXFP4 [*, N, 2048] u8 + e8m0 [*, N, 128] -> [N, 4096] fp64."""
    w = w.view(torch.uint8).reshape(-1, N, 2048)[e].cpu()
    s = s.view(torch.uint8).reshape(-1, N, 128)[e].cpu()
    t = E2M1
    v = torch.stack([t[(w & 15).long()], t[(w >> 4).long()]], -1).reshape(N, 4096)
    return v * torch.pow(2.0, s.double() - 127).repeat_interleave(32, 1)


def mxfp8(x):
    """CPU emulation of the kernels' MXFP8 rule on [R, 4096] fp64 (values are bf16)."""
    xb = x.reshape(x.shape[0], 128, 32).float()
    amax = xb.abs().amax(2)
    bits = (amax * (1.0 / 448.0)).view(torch.int32)
    se = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    se = se.clamp(max=253)
    inv = ((254 - se) << 23).view(torch.float32)
    q = (xb * inv[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).double()
    return (q * torch.pow(2.0, se.double() - 127)[..., None]).reshape(x.shape)


def mxfp4(x):
    """CPU emulation of an MXFP4 activation (E2M1 values, one power-of-two scale per 32, chosen so
    |x| / scale <= 6, round to nearest): the A4 numerics of the gfx950 path, for comparison."""
    xb = x.reshape(x.shape[0], 128, 32)
    amax = xb.abs().amax(2).clamp_min(1e-300)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
    y = (xb / scale[..., None]).clamp(-6, 6)
    idx = (y.abs()[..., None] - grid).abs().argmin(-1)
    return (torch.sign(y) * grid[idx] * scale[..., None]).reshape(x.shape)


REAL_LAYER = int(os.environ.get("REAL_LAYER", "-1"))
REAL_MODEL = os.environ.get(
    "REAL_MODEL", "/home/bangtliu/models/Muse-Spark-Open-816B-A42B-Quark-MXFP4")


def real_experts(w, layer):
    """Overwrite the fixture's expert weights in place with the real checkpoint's packed MXFP4
    experts of `layer` (the first experts, as many as the fixture holds). Activations and the
    routing table stay the fixture's."""
    import json as js
    import struct

    from safetensors import safe_open

    index = {}
    for f in sorted(Path(REAL_MODEL).glob("model-*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            for k in js.loads(fh.read(n)):
                index[k] = f
    handles = {}

    def get(name):
        f = index[name]
        if f not in handles:
            handles[f] = safe_open(str(f), "pt")
        return handles[f].get_tensor(name)

    pre = f"model.language_model.layers.{layer}.mlp.experts."
    for key, parts, suffix in (("w13", ("gate_proj", "up_proj"), "weight"),
                               ("w13_scale", ("gate_proj", "up_proj"), "weight_scale"),
                               ("w2", ("down_proj",), "weight"),
                               ("w2_scale", ("down_proj",), "weight_scale")):
        t = w[key]
        rows = 4096 * len(parts)
        flat = t.view(torch.uint8).reshape(-1, rows, t.numel() // (t.numel() // t.shape[-1]) )
        for e in range(flat.shape[0]):
            real = torch.cat([get(f"{pre}{e}.{p}.{suffix}") for p in parts], 0)
            assert real.dtype == torch.uint8 and real.shape == flat[e].shape, (
                key, real.dtype, real.shape, flat[e].shape)
            flat[e].copy_(real)
    print(json.dumps(dict(real_layer=layer, experts=int(flat.shape[0]),
                          shapes={k: list(w[k].shape) for k in ("w13", "w13_scale", "w2",
                                                                "w2_scale")})), flush=True)


def err(a, b, rows):
    a, b = a[rows].double(), b[rows].double()
    return dict(rel_l2=float((a - b).norm() / b.norm()),
                max_rel=float((a - b).abs().max() / b.abs().max()),
                max_abs=float((a - b).abs().max()))


def main():
    require_gpu0()
    out = []
    for context, left in [(8192, -1), (8192, 2047), (131072, -1), (131072, 2047)]:
        base, native_, *rest = setup(context, left)
        x, stream, w, m, _ = base["chained"][1].args
        if REAL_LAYER >= 0:
            real_experts(w, REAL_LAYER)
        res = {}
        off = {"tlx_experts.MX13": False, "tlx_experts.SPLIT13": False,
               "tlx_experts.FOLD13": False, "tlx_experts.FOLD2": False}
        on = {k: True for k in off}
        for name, flags in (("base", off), ("mx13", {**off, "tlx_experts.MX13": True}),
                             ("split13", {**on, "tlx_experts.FOLD13": False,
                                          "tlx_experts.FOLD2": False}),
                             ("fold13", {**on, "tlx_experts.FOLD2": False}),
                             ("fold2", on)):
            ws = with_flags(flags, lambda: tlx_chain.make_workspace("cuda", context=context,
                                                                    left=left))
            tlx_chain.prepare_peers(ws, *native_.partials, rank=0)
            with_flags(flags, lambda: tlx_chain.launch(x, stream, w, m, ws, rank=0))
            torch.cuda.synchronize()
            res[name] = (ws.s5.inter[0].clone(), ws.s5.output[0].clone())
            keep = ws
        s4 = keep.s4
        x4 = s4.pooled.clone()
        sorted_ids, experts, num_post = s4.sorted_ids, s4.expert_ids, s4.num_post
        # regular path: SGLang's fused MoE Triton GEMMs on the same table (rank 0)
        inter_r = torch.full((64, 4096), float("nan"), device="cuda", dtype=torch.bfloat16)
        y_r = torch.full_like(inter_r, float("nan"))
        cfg = native._pick_config(8, 8, 64, True)
        w13, s13 = w["w13"], w["w13_scale"]
        w2, s2 = w["w2"], w["w2_scale"]
        if w13.dim() == 4:
            w13, s13, w2, s2 = w13[0], s13[0], w2[0], s2[0]
        sid = sorted_ids.reshape(-1)[: sorted_ids.numel() // (4 if sorted_ids.dim() > 1 else 1)]
        native._launch_gemm(x4, w13, s13, inter_r, sorted_ids, experts, num_post,
                            4096, 4096, 64, 8, True, True, True, cfg)
        native._launch_gemm(inter_r, w2, s2, y_r, sorted_ids, experts, num_post,
                            4096, 4096, 64, 8, False, False, True, cfg)
        torch.cuda.synchronize()
        # FP64 reference
        srt = sorted_ids.reshape(-1).cpu()
        exp = experts.reshape(-1).cpu()
        nblk = int(num_post.reshape(-1)[0]) // 16
        xd = x4.double().cpu()
        inter64 = torch.zeros(64, 4096, dtype=torch.float64)
        emul = torch.zeros(64, 4096, dtype=torch.float64)
        xq = mxfp8(xd)
        xq2 = xq + mxfp8(xd - xq)  # two-term split: hi + MXFP8(residual)
        emul2 = torch.zeros(64, 4096, dtype=torch.float64)
        emul4 = torch.zeros(64, 4096, dtype=torch.float64)
        xq4 = mxfp4(xd)
        y2 = torch.zeros(64, 4096, dtype=torch.float64)
        y64 = torch.zeros(64, 4096, dtype=torch.float64)
        rows = []
        for b in range(nblk):
            e = int(exp[b])
            sl = [int(v) for v in srt[b * 16:(b + 1) * 16] if int(v) < 64]
            if not sl:
                continue
            g = deq(w13, s13, e, 8192)
            a = xd[[v // 8 for v in sl]]
            h = a @ g.T
            gate, up = h[:, :4096], h[:, 4096:]
            inter64[sl] = gate / (1 + torch.exp(-gate)) * up
            hq = xq[[v // 8 for v in sl]] @ g.T
            emul[sl] = hq[:, :4096] / (1 + torch.exp(-hq[:, :4096])) * hq[:, 4096:]
            h2 = xq2[[v // 8 for v in sl]] @ g.T
            emul2[sl] = h2[:, :4096] / (1 + torch.exp(-h2[:, :4096])) * h2[:, 4096:]
            h4 = xq4[[v // 8 for v in sl]] @ g.T
            emul4[sl] = h4[:, :4096] / (1 + torch.exp(-h4[:, :4096])) * h4[:, 4096:]
            ib = emul2[sl].float().bfloat16().double()
            y2[sl] = (mxfp8(ib) + mxfp8(ib - mxfp8(ib))) @ deq(w2, s2, e, 4096).T
            y64[sl] = inter64[sl] @ deq(w2, s2, e, 4096).T
            rows += sl
        row = dict(context=context, left=left, slots=len(rows), blocks=nblk,
                   meta=dict(w13=list(w13.shape), sorted=list(sorted_ids.shape)))
        row["regular"] = dict(w13=err(inter_r.cpu(), inter64, rows), w2=err(y_r.cpu(), y64, rows))
        row["emulated_mx13_w13"] = err(emul, inter64, rows)
        row["emulated_split_w13"] = err(emul2, inter64, rows)
        row["emulated_a4_w13"] = err(emul4, inter64, rows)
        row["emulated_split_w2_both"] = err(y2, y64, rows)
        row["mx13_vs_emulation_w13"] = err(res["mx13"][0].cpu(), emul, rows)
        for name, (inter, y) in res.items():
            row[name] = dict(w13=err(inter.cpu(), inter64, rows), w2=err(y.cpu(), y64, rows))
        bi = res["base"][0][rows].view(torch.int16)
        for name in ("split13", "fold13", "fold2", "mx13"):
            oi = res[name][0][rows].view(torch.int16)
            row[name + "_w13_elements_differing_from_base"] = int((oi != bi).sum())
        row["w13_elements"] = int(bi.numel())
        yb = res["base"][1][rows].view(torch.int16)
        for name in ("fold13", "fold2"):
            row[name + "_w2_elements_differing_from_base"] = int(
                (res[name][1][rows].view(torch.int16) != yb).sum())
        print(json.dumps(row), flush=True)
        out.append(row)


if __name__ == "__main__":
    main()

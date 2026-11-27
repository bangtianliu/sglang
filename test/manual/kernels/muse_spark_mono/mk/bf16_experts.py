"""S5 with BF16 expert weights (layers 0 and 61) on GPU0: numerics vs FP64 and timing vs E.

One rank's 64 local experts, an 8-token verify step whose routing puts E local experts on this
rank (one 16-slot block each). Variants: TLX expert_phase WBF16 at (K_STEP, DEPTH) and the
native BF16 muse_moe_triton GEMMs on the same table. Timing is W13 + W2 launches only:
  warm: the same experts back to back;
  cold: a 1 GiB scrub between launches and a rotating set of experts (weights not in any cache).
Env: REAL_LAYER=0|61 loads that layer's experts 0..63 from REAL_MODEL (numerics run only);
ES (default 2,4,6,8,10,12,14,16), VARIANTS (default 256x2,128x3), REPS (default 30).
"""

import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import triton  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_experts  # noqa: E402
from sglang.srt.rocm_muse_spark import muse_moe_triton as native  # noqa: E402

T, TOPK, LOCAL, H = 8, 8, 64, 4096
REAL_LAYER = int(os.environ.get("REAL_LAYER", "-1"))
REAL_MODEL = os.environ.get("REAL_MODEL", "/models/Muse-Spark-Open-816B-A42B-Quark-MXFP4")
ES = [int(v) for v in os.environ.get("ES", "2,4,6,8,10,12,14,16").split(",")]
VARIANTS = [tuple(int(p) for p in v.split("x"))
            for v in os.environ.get("VARIANTS", "256x2,128x3").split(",")]
REPS = int(os.environ.get("REPS", "30"))


def routing(E, base, gen):
    """topk_ids [8, 8]: E local experts base..base+E-1 (all used), the rest non-local."""
    local = list(range(base, base + E))
    ids = [[] for _ in range(T)]
    # every local expert gets at least one token; then about one more route per expert
    order = torch.randperm(T * E, generator=gen).tolist()
    want = min(T * TOPK, 2 * E)
    for e in local:
        t = min(range(T), key=lambda t: len(ids[t]) + 0.01 * ((t * 7 + e) % T))
        ids[t].append(e)
    for k in order:
        if sum(len(r) for r in ids) >= want:
            break
        t, e = k // E, local[k % E]
        if e not in ids[t] and len(ids[t]) < TOPK:
            ids[t].append(e)
    for t in range(T):
        pool = torch.randperm(256 - LOCAL, generator=gen).tolist()
        for p in pool:
            if len(ids[t]) == TOPK:
                break
            ids[t].append(LOCAL + p)
    return torch.tensor(ids, dtype=torch.int32, device="cuda")


def table(topk_ids):
    s, e, n = native.moe_align_block_size_torch(topk_ids, LOCAL, 0, 16)
    return s.contiguous(), e.contiguous(), n.contiguous()


def tlx_s5(x, tab, w13, w2, inter, out, k_step, depth, grid, wpe):
    sorted_ids, experts, num_post = tab
    status = torch.zeros(1728, dtype=torch.int32, device="cuda")
    failure = torch.zeros(2, dtype=torch.int32, device="cuda")
    common = dict(GRID=grid, WBF16=True, K_STEP=k_step, DEPTH=depth, num_warps=4,
                  waves_per_eu=wpe, enable_fp_fusion=False)
    tlx_experts.expert_phase[(grid,)](x, sorted_ids, experts, num_post, w13, w13, inter, status,
                                      failure, FIRST=True, **common)
    tlx_experts.expert_phase[(grid,)](inter, sorted_ids, experts, num_post, w2, w2, out, status,
                                      failure, FIRST=False, CROSS_LAUNCH_A=True, **common)
    return failure


def native_s5(x, tab, w13, w2, inter, out):
    # _launch_gemm's N is the output width (after SwiGLU for W13), not the weight rows
    assert inter.shape == out.shape == (T * TOPK, H), (inter.shape, out.shape)
    assert w13.shape == (LOCAL, 2 * H, H) and w2.shape == (LOCAL, H, H), (w13.shape, w2.shape)
    sorted_ids, experts, num_post = tab
    cfg = native._pick_config(T, TOPK, LOCAL, False)
    native._launch_gemm(x, w13, None, inter, sorted_ids, experts, num_post,
                        H, H, T * TOPK, TOPK, True, True, False, cfg)
    native._launch_gemm(inter, w2, None, out, sorted_ids, experts, num_post,
                        H, H, T * TOPK, TOPK, False, False, False, cfg)


def real_experts(w13, w2, layer):
    from safetensors import safe_open
    index = {}
    for f in sorted(Path(REAL_MODEL).glob("model-*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            for k in json.loads(fh.read(n)):
                index[k] = f
    handles = {}

    def get(name):
        f = index[name]
        if f not in handles:
            handles[f] = safe_open(str(f), "pt")
        t = handles[f].get_tensor(name)
        assert t.dtype == torch.bfloat16 and t.shape == (H, H), (name, t.dtype, t.shape)
        return t

    pre = f"model.language_model.layers.{layer}.mlp.experts."
    for e in range(LOCAL):
        w13[e, :H].copy_(get(f"{pre}{e}.gate_proj.weight"))
        w13[e, H:].copy_(get(f"{pre}{e}.up_proj.weight"))
        w2[e].copy_(get(f"{pre}{e}.down_proj.weight"))


def err(a, b, rows):
    a, b = a[rows].double(), b[rows].double()
    return dict(rel_l2=float((a - b).norm() / b.norm()),
                max_rel=float((a - b).abs().max() / b.abs().max()))


def fp64_reference(x, tab, w13, w2):
    sorted_ids, experts, num_post = (v.cpu() for v in tab)
    inter = torch.zeros(T * TOPK, H, dtype=torch.float64)
    y = torch.zeros_like(inter)
    y_rounded = torch.zeros_like(inter)
    rows = []
    for b in range(int(num_post[0]) // 16):
        sl = [int(v) for v in sorted_ids[b * 16:(b + 1) * 16] if int(v) < T * TOPK]
        if not sl:
            continue
        e = int(experts[b])
        h = x[[v // TOPK for v in sl]].double().cpu() @ w13[e].double().cpu().T
        inter[sl] = h[:, :H] / (1 + torch.exp(-h[:, :H])) * h[:, H:]
        w2e = w2[e].double().cpu().T
        y[sl] = inter[sl] @ w2e
        y_rounded[sl] = inter[sl].bfloat16().double() @ w2e
        rows += sl
    return inter, y, y_rounded, rows


def numerics(x, w13, w2, gen):
    tab = table(routing(8, 0, gen))
    ref_i, ref_y, ref_yr, rows = fp64_reference(x, tab, w13, w2)
    row = dict(kind="numerics", real_layer=REAL_LAYER, E=8, slots=len(rows))
    grid = tlx_experts.expert_launch_grid(x.device)
    for k_step, depth in VARIANTS:
        inter = torch.full((T * TOPK, H), float("nan"), device="cuda", dtype=torch.bfloat16)
        out = torch.full_like(inter, float("nan"))
        fail = tlx_s5(x, tab, w13, w2, inter, out, k_step, depth, grid, 1)
        torch.cuda.synchronize()
        row[f"tlx_{k_step}x{depth}"] = dict(
            failure=fail.tolist(), w13=err(inter.cpu(), ref_i, rows), w2=err(out.cpu(), ref_y, rows),
            w2_vs_rounded_inter=err(out.cpu(), ref_yr, rows))
        keep = (inter.clone(), out.clone())
    inter = torch.full((T * TOPK, H), float("nan"), device="cuda", dtype=torch.bfloat16)
    out = torch.full_like(inter, float("nan"))
    native_s5(x, tab, w13, w2, inter, out)
    torch.cuda.synchronize()
    row["native"] = dict(w13=err(inter.cpu(), ref_i, rows), w2=err(out.cpu(), ref_y, rows))
    row["tlx_last_vs_native"] = dict(w13=err(keep[0].cpu(), inter.cpu().double(), rows),
                                     w2=err(keep[1].cpu(), out.cpu().double(), rows))
    print(json.dumps(row), flush=True)


def timed(fn, cold, scrub):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for r in range(REPS + 3):
        if cold:
            scrub.add_(1)
        torch.cuda.synchronize()
        start.record()
        fn(r)
        end.record()
        torch.cuda.synchronize()
        if r >= 3:
            times.append(start.elapsed_time(end) * 1000)
    times.sort()
    return dict(median_us=round(times[len(times) // 2], 1), min_us=round(times[0], 1))


def timing(x, w13, w2, gen):
    grid = tlx_experts.expert_launch_grid(x.device)
    scrub = torch.zeros(256 << 20, dtype=torch.float32, device="cuda")  # 1 GiB
    inter = torch.zeros((T * TOPK, H), device="cuda", dtype=torch.bfloat16)
    out = torch.zeros_like(inter)
    for E in ES:
        sets = max(1, LOCAL // E)
        tabs = [table(routing(E, (i * E) % (LOCAL - E + 1), gen)) for i in range(sets)]
        row = dict(kind="timing", E=E, grid=grid, weight_MB=round(E * 3 * H * H * 2 / 1e6, 1))
        for k_step, depth in VARIANTS:
            for wpe in (1, 2):
                name = f"tlx_{k_step}x{depth}_wpe{wpe}"
                try:
                    tlx_s5(x, tabs[0], w13, w2, inter, out, k_step, depth, grid, wpe)
                    row[name] = dict(
                        warm=timed(lambda r: tlx_s5(x, tabs[0], w13, w2, inter, out, k_step,
                                                    depth, grid, wpe), False, scrub),
                        cold=timed(lambda r: tlx_s5(x, tabs[r % sets], w13, w2, inter, out,
                                                    k_step, depth, grid, wpe), True, scrub))
                except Exception as exc:  # a variant over the LDS limit fails to compile
                    row[name] = dict(error=str(exc).splitlines()[0][:200])
        row["native"] = dict(
            warm=timed(lambda r: native_s5(x, tabs[0], w13, w2, inter, out), False, scrub),
            cold=timed(lambda r: native_s5(x, tabs[r % sets], w13, w2, inter, out), True, scrub))
        for name, v in row.items():
            if isinstance(v, dict) and "cold" in v:
                v["cold_TBps"] = round(row["weight_MB"] / v["cold"]["median_us"], 2)
        print(json.dumps(row), flush=True)


def main():
    require_gpu0()
    gen = torch.Generator().manual_seed(1234)
    torch.manual_seed(1234)
    w13 = torch.empty((LOCAL, 2 * H, H), device="cuda", dtype=torch.bfloat16)
    w2 = torch.empty((LOCAL, H, H), device="cuda", dtype=torch.bfloat16)
    if REAL_LAYER >= 0:
        real_experts(w13, w2, REAL_LAYER)
    else:
        w13.normal_(0, 0.02)
        w2.normal_(0, 0.02)
    x = (torch.randn((T, H), device="cuda") * 0.5).bfloat16()
    print(json.dumps(dict(kind="meta", device=torch.cuda.get_device_name(),
                          cus=torch.cuda.get_device_properties(0).multi_processor_count,
                          triton=triton.__version__, real_layer=REAL_LAYER)), flush=True)
    numerics(x, w13, w2, gen)
    if REAL_LAYER < 0:
        timing(x, w13, w2, gen)


if __name__ == "__main__":
    main()

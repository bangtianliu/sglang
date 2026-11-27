import sys, json, functools, types, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import torch
from bench_chained47 import setup
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain, tlx_s3fused
base, native, *rest = setup(8192, -1)
x, stream, w, m, _ = base["chained"][1].args
ws = tlx_chain.make_workspace("cuda", context=8192, left=-1)
st = tlx_s3fused.State("cuda"); trace = torch.zeros(272 * 8, device="cuda", dtype=torch.int64)
def at(x_, gate, o_proj, stream_, pre, alpha, beta, *, rank, workspace, store_reduced, split_norm, output, failure, **_):
    return tlx_s3fused.tail(x_, gate, o_proj, stream_, pre, alpha, beta, rank=rank, ws=workspace, st=st, output=output, failure=failure, split_norm=split_norm, store_reduced=store_reduced, trace=trace, s4=S4ARGS, sleep4=int(os.environ.get('SLEEP4', '8')), pre4=os.environ.get('PRE4', '1') == '1', s4tiles=int(os.environ.get('S4TILES', '72')), x4wt=os.environ.get('X4WT', '1') == '1', warm=os.environ.get('WARM') == '1', nb4=4, s4x=os.environ.get('S4X') == '1', rows4=ROWS4, head_sleep=int(os.environ.get('HS', '0')), warm4=os.environ.get('WARM4') == '1')
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_route
import os
S4ARGS = (w['joint_router_pooler'], ws.s4.projection, ws.s4.producer) if os.environ.get('S4') else None
ROWS4 = (w['bias'], w['route_norm'], ws.s4.scores, ws.s4.ids, ws.s4.pooled, ws.s4.rows_ready) if os.environ.get('ROWS4') else None
def rr(x4, weight, bias, norm, *, rank=0, workspace=None, failure=None, **_):
    W = workspace; failure = W.local_failure if failure is None else failure
    if ROWS4 is None: tlx_route.finish_rows[(8,)](W.projection, bias, norm, W.scores, W.ids, W.pooled, W.producer, W.rows_ready, failure, num_warps=4, enable_fp_fusion=False)
    tlx_route.slots[(64,)](W.ids, W.sorted_ids, W.expert_ids, W.num_post, W.rows_ready, W.state.status, failure, LOCAL_START=0, GRID=W.state.grid, num_warps=4, enable_fp_fusion=False)
for it in range(5):
    if it == 1: tlx_s3fused.refresh_code_range(st); print('CODE', st.code)
    tlx_chain.prepare_peers(ws, *native.partials, rank=0); trace.zero_(); torch.cuda.synchronize()
    old = tlx_chain.tlx_attn_tail; old4 = tlx_chain.tlx_route; tlx_chain.tlx_attn_tail = types.SimpleNamespace(attention_tail=at)
    if S4ARGS is not None: tlx_chain.tlx_route = types.SimpleNamespace(routing=rr)
    try: tlx_chain.launch(x, stream, w, m, ws, rank=0)
    finally: tlx_chain.tlx_attn_tail = old; tlx_chain.tlx_route = old4
    torch.cuda.synchronize()
T = trace.view(272, 8).double(); t = T[:128]
base0 = t[:, 0].min()
us = lambda col: (t[:, col] - base0) / 100.0
def stat(v): v = v.sort().values; return [round(float(v[0]), 2), round(float(v[len(v)//2]), 2), round(float(v[-1]), 2)]
print("tile start", stat(us(0)), "tile done", stat(us(1)), "push+arrive", stat(us(2)))
rw = t[:, 5] > 0
s4 = (t[:, 7] > 0) & (t[:, 5] == 0)
print("s4 tiles", int(s4.sum()), "s4 done", stat(us(7)[s4]) if int(s4.sum()) else None)
print("row workers", int(rw.sum()), "wait done", stat(us(4)[rw]), "gather done", stat(us(5)[rw]), "norm done", stat(us(6)[rw]), "drain+signal done", stat(us(7)[rw]))
for dc in tlx_s3fused.s3_fused.device_caches.values():
    for key, k in dc[0].items():
        print("KMETA", "S4" if "True" in str(key)[-400:] else "?", "vgpr", k.n_regs, "spills", k.n_spills, "lds", k.metadata.shared, "scratch", getattr(k.metadata, "global_scratch_size", None))
for dc in tlx_s3fused.s3_fused.device_caches.values():
    for key, k in dc[0].items():
        asm = k.asm["amdgcn"]
        import re
        print("KISA vgpr", k.n_regs, "sgpr", getattr(k, "n_sgprs", None), "loads", len(re.findall(r"global_load|buffer_load", asm)),
              "wait_loadcnt", len(re.findall(r"s_wait_loadcnt", asm)), "lines", asm.count("\n"),
              "waves_per_eu", re.findall(r"amdhsa_next_free_vgpr\s+\d+", asm)[:1], re.findall(r'"amdgpu-waves-per-eu"[^\n]*', k.asm.get("llir", ""))[:1])

if os.environ.get("ROWS4"):
    q = T[128:]
    b0 = t[:, 0].min()
    u = lambda col, m: ((q[:, col] - b0) / 100.0)[m]
    half0 = q[:, 1] > 0
    rowsm = q[:, 4] > 0
    print("S4 half0 blocks", int(half0.sum()), "L+R done", stat(u(1, half0)), "stored+arrived", stat(u(2, half0)))
    print("S4 row workers", int(rowsm.sum()), "wait done", stat(u(4, rowsm)), "rows end", stat(u(5, rowsm)))

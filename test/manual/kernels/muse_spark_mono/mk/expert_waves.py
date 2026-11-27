"""MK.10a (GPU0): W13/W2 phase time vs number of active expert blocks (wave-quantization test).
Runs a real layer to build the routing table, then re-launches each expert phase alone with
NumPost = 16 * nb, nb = 1..10, rotating 3 weight copies. Prints occupancy and medians."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402
from layer_case import make  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402
from reference_archive import gpu  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain, tlx_experts  # noqa: E402

captured = []
_orig = tlx_experts.s5


def spy(*a, **k):
    r = _orig(*a, **k)
    captured.append((a, k))
    return r


def main():
    require_gpu0()
    old = gpu(torch.load(Path("/home/bangtliu/coord/phase44-baseline") / "layer-7-8192--1.pt",
                         weights_only=True)["rows"][0])
    tlx_experts.s5 = spy
    tlx_chain.tlx_experts.s5 = spy
    copies = []
    for _ in range(3):
        f, w, m = make(7, 8192, -1, 0)
        ws = tlx_chain.make_workspace("cuda", context=8192, left=-1)
        tlx_chain.prepare_peers(ws, old["partial3"], old["partial5"], rank=0)
        tlx_chain.launch(f.x, f.stream, w, m, ws, rank=0)
        copies.append((f, w, m, ws))
    torch.cuda.synchronize()
    print(json.dumps(dict(admitted=list(copies[0][3].s5.state.admitted.values())[0])), flush=True)
    # rebuild the phase launches from the captured s5 arguments (same as tlx_experts.s5)
    import inspect
    src = inspect.getsource(_orig)
    assert "phases = [" in src
    graphs = {}
    keep = []
    for phase in ("w13", "w2"):
        for ci, (a, k) in enumerate(captured[:3]):
            fa, fk = phase_args(a, k, phase)
            tlx_experts.expert_phase[(1728,)](*fa, **fk)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                tlx_experts.expert_phase[(1728,)](*fa, **fk)
            graphs[(phase, ci)] = g
            keep.append((fa, fk))
    for phase in ("w13", "w2"):
        for nb in list(range(1, 13)):
            ts = []
            for rep in range(10):
                for ci, (a, k) in enumerate(captured[:3]):
                    npost = a[3][2]
                    saved = npost.clone()
                    npost.fill_(16 * nb)
                    torch.cuda.synchronize()
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    graphs[(phase, ci)].replay()
                    e.record()
                    torch.cuda.synchronize()
                    npost.copy_(saved)
                    ts.append(s.elapsed_time(e) * 1000)
            ts.sort()
            print(json.dumps(dict(phase=phase, blocks=nb, tasks=nb * 64, us=round(ts[len(ts) // 2], 1))), flush=True)


def phase_args(a, k, phase):
    x, ids, scores, table, weights, gamma, ws = a[:7]
    s = ws.state
    failure = ws.local_failure if k.get("failure") is None else k["failure"]
    sorted_ids, experts, num_post = table
    common = dict(GRID=1728, num_warps=4, waves_per_eu=9, enable_fp_fusion=False, SCALE_ASYNC=ws.scale_async)
    rank = k["rank"]
    if phase == "w13":
        args = (x, sorted_ids, experts, num_post, weights["w13"], weights["w13_scale"], ws.inter[rank], s.status, failure)
        kk = dict(**common, FIRST=True, SKIP=-1, Seed=None, WEIGHT_CACHE="")
        if ws.wide_w13:
            kk.update(WIDE=True, SCALE_ASYNC=True, waves_per_eu=2)
    else:
        args = (ws.inter[rank], sorted_ids, experts, num_post, weights["w2"], weights["w2_scale"], ws.output[rank], s.status, failure)
        kk = dict(**common, FIRST=False, Seed=None, WEIGHT_CACHE="", CROSS_LAUNCH_A=True)
        if ws.wide_w2:
            kk.update(WIDE=True, SCALE_ASYNC=True, waves_per_eu=2)
    return args, kk


if __name__ == "__main__":
    main()

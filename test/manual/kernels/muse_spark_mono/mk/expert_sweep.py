"""MK.14f (GPU0): expert-phase (S5: W13, W2, sum/AR) cold time vs. the number of local
experts a rank receives, for several expert configs.

Routing changes every decode step and the slowest rank sets the layer time, so the
expert config should be chosen over a range of local-expert counts, not the one
archived routing. Synthetic routings: E distinct local experts (rank 0 owns 0..63),
about E/4 of them hit twice; the rest of the 64 routes go to other ranks. The slot
table is built on the host with slots_body's algorithm.

usage: expert_sweep.py [--configs cur,d2,w2,d2w2] [--experts 8,12,16,20,24] [--rounds 10]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from bench_chained47 import setup  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (  # noqa: E402
    tlx_chain,
    tlx_experts,
)

CONFIGS = {
    "cur": {},
    "d2": {"DEPTH2": 2},
    "w2": {"WPE13": 2},
    "d2w2": {"DEPTH2": 2, "WPE13": 2},
}


def routing(E, seed):
    """ids (8 tokens x 8 routes, distinct per token) with E distinct local experts."""
    rng = random.Random(seed)
    local = rng.sample(range(64), E)
    hits = local + rng.sample(local, E // 4)  # some experts hit twice
    ids = [[None] * 8 for _ in range(8)]
    slots = [(t, j) for t in range(8) for j in range(8)]
    rng.shuffle(slots)
    used = {t: set() for t in range(8)}
    for e in hits:
        for k, (t, j) in enumerate(slots):
            if e not in used[t]:
                ids[t][j] = e
                used[t].add(e)
                slots.pop(k)
                break
    for t, j in slots:
        e = rng.choice([x for x in range(64, 256) if x not in used[t]])
        ids[t][j] = e
        used[t].add(e)
    return torch.tensor(ids, dtype=torch.int32)


def table(ids, rank=0):
    """Host copy of tlx_route.slots_body: Sorted (64 blocks x 16), Experts, NumPost."""
    flat = ids.view(-1).tolist()
    sorted_ = [64] * 1024
    experts = [-1] * 64
    block = 0
    for e in range(64):
        hits = [j for j, v in enumerate(flat) if v - rank * 64 == e]
        if not hits:
            continue
        for b in range((len(hits) + 15) // 16):
            experts[block] = e
            for d, j in enumerate(hits[b * 16:(b + 1) * 16]):
                sorted_[block * 16 + d] = j
            block += 1
    return (torch.tensor(sorted_, dtype=torch.int32), torch.tensor(experts, dtype=torch.int32),
            torch.tensor([block * 16], dtype=torch.int32))


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="cur,d2,w2,d2w2")
    p.add_argument("--experts", default="8,12,16,20,24")
    p.add_argument("--rounds", type=int, default=10)
    a = p.parse_args()
    names = a.configs.split(",")
    base, native, *rest = setup(8192, -1)
    x, stream, w, m, _ = base["chained"][1].args
    ws = tlx_chain.make_workspace("cuda", context=8192, left=-1)
    tlx_chain.prepare_peers(ws, *native.partials, rank=0)
    tlx_chain.launch(x, stream, w, m, ws, rank=0)
    torch.cuda.synchronize()
    keys = ("w13", "w13_scale", "w2", "w2_scale")
    copies = [w] + [dict(w, **{k: w[k].clone() for k in keys}) for _ in range(2)]
    out = torch.empty_like(ws.buffers["x6"])
    retained = [base, native, rest, copies]

    def prep():
        tlx_chain.prepare_peers(ws, *native.partials, rank=0)

    for E in map(int, a.experts.split(",")):
        ids = routing(E, 1000 + E).cuda()
        tab = tuple(t.cuda() for t in table(ids.cpu()))
        retained.append((ids, tab))
        graphs = {}
        for name in names:
            saved = {k: getattr(tlx_experts, k) for k in CONFIGS[name]}
            for k, v in CONFIGS[name].items():
                setattr(tlx_experts, k, v)
            try:
                for ci, wc in enumerate(copies):
                    def run(wc=wc):
                        tlx_experts.s5(ws.s4.pooled, ids, ws.s4.scores, tab, wc, w["gamma5"],
                                       ws.s5, rank=0, output=out)
                    prep()
                    run()
                    torch.cuda.synchronize()
                    assert int(ws.failure[0]) == 0, ws.failure.tolist()
                    prep()
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        run()
                    graphs[(name, ci)] = g
                    retained.append(g)
            finally:
                for k, v in saved.items():
                    setattr(tlx_experts, k, v)
        times = {n: [] for n in names}
        for r in range(a.rounds):
            order = names if r % 2 == 0 else names[::-1]
            for ci in range(len(copies)):
                for name in order:
                    prep()
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    graphs[(name, ci)].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    assert int(ws.failure[0]) == 0
                    times[name].append(e0.elapsed_time(e1) * 1000)
        med = {n: round(sorted(t)[len(t) // 2], 2) for n, t in times.items()}
        print(json.dumps(dict(experts=E, tasks=int(tab[2]) // 16 * 64, s5_us=med,
                              ratio={n: round(med[n] / med[names[0]], 4) for n in names[1:]})),
              flush=True)


if __name__ == "__main__":
    main()

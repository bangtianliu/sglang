"""MK.91/92: the replicated-expert assignment in the slots kernel, alone on GPU0.

1. Checks the kernel's tables (Mine, Experts, Sorted, NumPost) against a Python reference of
   the rule on random and skewed routings, for all four ranks, and that the four ranks'
   assignments partition the routes.
2. Times the standalone slots kernel (graph of 200 launches) with and without the rule.
env: N (routings per kind, default 200), MODES (comma list of tlx_route.REPL values to time).
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_route  # noqa: E402


def reference(ids, masks, passes=3):
    """ids: 64 global expert ids (8 rows x 8); masks[a][k]: see tlx_route. Returns per-rank
    (mine[64], blocks), blocks = [(local expert index, [route indices])] in block order, and
    the busiest-rank block counts (levelled, EP4, ideal)."""
    bases = tlx_route.repl_bases(masks)

    def grp(e):
        a, ep = e >> 6, e & 63
        for k in range(3):
            if masks[a][k] >> ep & 1:
                return k
        return 3

    present = sorted(e for e in set(ids) if 0 <= e < 256)  # other ids are not routes
    g = [[0] * 3 for _ in range(4)]
    load = [0] * 4
    for e in present:
        load[e >> 6] += 1
        if grp(e) < 3:
            g[e >> 6][grp(e)] += 1
    ep4 = max(load)
    m = [[0] * 3 for _ in range(4)]
    for _ in range(passes):
        for a in range(4):
            for k in range(3):
                t = (a + 1 + k) & 3
                d = load[a] - load[t]
                h = d >> 1 if d >= 0 else -((-d) >> 1)
                mv = max(min(g[a][k] - m[a][k], h), -m[a][k])
                m[a][k] += mv
                load[a] -= mv
                load[t] += mv
    place = {}
    seen = [[0] * 3 for _ in range(4)]
    for e in present:  # ascending id = ascending bit inside each group
        a, ep = e >> 6, e & 63
        k = grp(e)
        moved = k < 3 and seen[a][k] < m[a][k]
        if k < 3:
            seen[a][k] += 1
        if moved:
            t = (a + 1 + k) & 3
            slot = bases[t][k] + bin(masks[a][k] & ((1 << ep) - 1)).count("1")
            place[e] = (t, 64 + slot, k, 1)
        else:
            place[e] = (a, ep, 0, 0)
    out = []
    for r in range(4):
        mine = [1 if e in place and place[e][0] == r else 0 for e in ids]
        exps = sorted((e for e in present if place[e][0] == r),
                      key=lambda e: (place[e][3], place[e][2], e & 63))
        blocks = [(place[e][1], [i for i, x in enumerate(ids) if x == e]) for e in exps]
        out.append((mine, blocks))
    return out, max(load), ep4, -(-len(present) // 4)


def tables(rng):
    """(name, masks): every expert copied; a random quarter; the hot low ids of each rank."""
    full = tlx_route.repl_masks_full()
    quarter, hot = [], []
    for a in range(4):
        ks = [[0, 0, 0], [0, 0, 0]]
        for which, chosen in enumerate((rng.sample(range(64), 16), range(16))):
            for n, ep in enumerate(chosen):
                ks[which][n % 3] |= 1 << ep
        quarter.append(ks[0])
        hot.append(ks[1])
    none = [[0, 0, 0] for _ in range(4)]
    return [("full", full), ("quarter", quarter), ("hot16", hot), ("none", none)]


def routing(rng, kind):
    if kind == "uniform":
        w = [1.0] * 256
    elif kind == "zipf":
        w = [1 / (i + 1) for i in range(256)]
        rng.shuffle(w)
    elif kind == "garbage":  # a few rows carry ids a non-finite router row produces
        w = [1.0] * 256
    else:  # clustered: hot experts are the low ids (all on rank 0)
        w = [1 / (i + 1) for i in range(256)]
    ids = []
    for _ in range(8):
        row = set()
        while len(row) < 8:
            row.add(rng.choices(range(256), w)[0])
        row = list(row)
        rng.shuffle(row)
        ids += row
    if kind == "garbage":
        for r in rng.sample(range(8), rng.randint(1, 8)):
            v = rng.choice([2147483647, -1, 256, 1000, -2147483648, 2147483647 - 63])
            for sl in range(8):
                ids[r * 8 + sl] = v if rng.random() < 0.8 else rng.choice([2147483647, -1, 300, 255 + 64])
    return ids


def run_slots(ws, ids_t, rank, mode, masks=None):
    args = (ids_t, ws.sorted_ids, ws.expert_ids, ws.num_post, ws.rows_ready, ws.state.status,
            ws.local_failure)
    kw = dict(LOCAL_START=rank * 64, GRID=ws.state.grid, num_warps=4, enable_fp_fusion=False)
    if mode:
        kw.update(Mine=tlx_route.repl_mine(ws.sorted_ids),
                  **tlx_route.repl_operands(rank, masks))
    tlx_route.slots[(64,)](*args, **kw)


def main():
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    ws = tlx_route.make_workspace("cuda")
    ws.rows_ready.fill_(1)
    ws.local_failure.zero_()
    ids_t = torch.zeros(64, device="cuda", dtype=torch.int32)
    mine_t = tlx_route.repl_mine(ws.sorted_ids)
    n = int(os.environ.get("N", "200"))
    if n == 0:  # code-size report only
        run_slots(ws, ids_t, 2, 0)
        run_slots(ws, ids_t, 2, 1)
        torch.cuda.synchronize()
        code_report()
        return
    rng = random.Random(0)
    bad = 0
    oob = 0
    stats = {}
    keep = []
    for tname, masks in tables(rng):
        for kind in ("uniform", "zipf", "clustered", "garbage"):
            tot = [0, 0, 0, 0]
            for _ in range(n):
                ids = routing(rng, kind)
                ref, lv, ep4, ideal = reference(ids, masks)
                tot = [tot[0] + lv, tot[1] + ep4, tot[2] + ideal, tot[3] + 1]
                ids_t.copy_(torch.tensor(ids, dtype=torch.int32))
                cover = [0] * 64
                for r in range(4):
                    ws.sorted_ids.fill_(-7)
                    ws.expert_ids.fill_(-7)
                    run_slots(ws, ids_t, r, 1, masks)
                    torch.cuda.synchronize()
                    mine = mine_t.tolist()
                    nb = int(ws.num_post.item()) // 16
                    experts = ws.expert_ids[:nb].tolist()
                    srt = ws.sorted_ids.view(64, 16)[:nb].tolist()
                    want_mine, want_blocks = ref[r]
                    got_blocks = [(e, [x for x in row if x < 64]) for e, row in zip(experts, srt)]
                    hosted_r = tlx_route.repl_bases(masks)[r][3]
                    oob += any(not 0 <= e < 64 + hosted_r for e in experts)
                    ok = (mine == want_mine and got_blocks == want_blocks
                          and all(x == -1 for x in ws.expert_ids[nb:].tolist()))
                    if not ok and bad < 3:
                        print(json.dumps(dict(table=tname, kind=kind, rank=r, ids=ids, mine=mine,
                                              want=want_mine, got=got_blocks,
                                              want_blocks=want_blocks)), flush=True)
                    bad += not ok
                    cover = [c + x for c, x in zip(cover, mine)]
                bad += cover != [1 if 0 <= e < 256 else 0 for e in ids]
            stats[f"{tname}/{kind}"] = dict(levelled=round(tot[0] / tot[3], 2),
                                            ep4=round(tot[1] / tot[3], 2),
                                            ideal=round(tot[2] / tot[3], 2))
    print(json.dumps(dict(check="kernel vs reference", routings=16 * n, ranks=4, mismatches=bad, out_of_range=oob,
                          busiest_rank_blocks=stats)), flush=True)
    full_tab = tlx_route.repl_masks_full()
    ids_t.copy_(torch.tensor(routing(random.Random(1), "uniform"), dtype=torch.int32))
    for mode in [int(x) for x in os.environ.get("MODES", "0,1").split(",")]:
        run_slots(ws, ids_t, 2, mode, full_tab)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(200):
                run_slots(ws, ids_t, 2, mode, full_tab)
        best = 1e9
        for _ in range(20):
            torch.cuda.synchronize()
            t = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t) / 200 * 1e6)
        print(json.dumps(dict(mode=mode, slots_us=round(best, 2))), flush=True)
        del g


def code_report():
    """Instruction counts of every compiled variant of the standalone slots kernel."""
    import collections
    import re

    for dc in tlx_route.slots.device_caches.values():
        for k_ in dc[0].values():
            asm = k_.asm.get("amdgcn", "")
            c = collections.Counter()
            for line in asm.splitlines():
                m = re.match(r"\s+([a-z_0-9]+)\s", line)
                if m:
                    c[m.group(1)] += 1
            top = {k: v for k, v in c.most_common(14)}
            print(json.dumps(dict(code="slots", instructions=sum(c.values()),
                                  readlane=c["v_readlane_b32"], writelane=c["v_writelane_b32"],
                                  cselect=c["s_cselect_b32"], delay=c["s_delay_alu"],
                                  bcnt=c["v_bcnt_u32_b32"], top=top)), flush=True)
            if os.environ.get("DUMP"):
                Path(os.environ["DUMP"]).mkdir(parents=True, exist_ok=True)
                (Path(os.environ["DUMP"]) / f"slots-{sum(c.values())}.s").write_text(asm)


if __name__ == "__main__":
    main()

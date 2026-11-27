"""Real-checkpoint routing counts -> placement value, plus cross-workload transfer.

usage: eplb_real_counts.py counts_A.json [counts_B.json]
counts file: {"samples": n, "freq": [layer][256]} (layers 1..60 filled, from MUSE_ROUTE_STATS_FILE).
Per layer: concentration (top-8/16/32 share), EP4 home-rank mass imbalance, and the simulated busiest
rank (8 rows x top-8 sampled from the layer's own frequencies) for EP4 vs place() at budgets
8/16/32. With B: masks from A scored on B's frequencies (placement transfer) vs B's own oracle."""
import json, sys, random, statistics as st
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from eplb_place import place, owners_from_masks, place_model
from eplb_placement_study import min_max, sample_step

STEPS, BUDGETS = 200, (8, 16, 32)


def load(p):
    d = json.load(open(p))
    return {l: f for l, f in enumerate(d["freq"]) if sum(f) > 0}, d.get("samples")


def busy(freq, masks, seed):
    t = sum(freq); pop = [f / t for f in freq]; own = owners_from_masks(masks); r = random.Random(seed)
    return st.mean(min_max(sorted(set().union(*sample_step(r, pop, 8, 0))), own) for _ in range(STEPS))


def summary(name, F):
    print(f"## {name}: {len(F)} layers, {sum(F[next(iter(F))])} routes per layer")
    cov, imb = {8: [], 16: [], 32: []}, []
    for f in F.values():
        s = sorted(f, reverse=True); t = sum(f)
        for k in cov: cov[k].append(sum(s[:k]) / t)
        home = [sum(f[64 * r:64 * r + 64]) / t for r in range(4)]
        imb.append(max(home) / 0.25)
    print("  top-k share (mean over layers): " + "  ".join(f"top{k} {st.mean(v):.2f}" for k, v in cov.items()))
    print(f"  busiest home rank mass / even: mean {st.mean(imb):.2f}  max {max(imb):.2f}")


def gains(F, G=None, layers=None):
    G = G or F
    rows = []
    for l in layers or sorted(F):
        b0 = busy(G[l], place(G[l], 0), l)
        r = [busy(G[l], place(F[l], b), l) / b0 for b in BUDGETS]
        rows.append((l, b0, r))
    return rows


def main():
    A, _ = load(sys.argv[1]); summary("A", A)
    ra = gains(A)
    print("  busiest rank vs EP4, own placement:  " + "  ".join(
        f"b{b} {st.mean(r[2][i] for r in ra):.3f}" for i, b in enumerate(BUDGETS)))
    worst = sorted(ra, key=lambda r: r[2][1])[:5]
    print("  most-helped layers (b16): " + ", ".join(f"L{l} {r[1]:.2f}" for l, _, r in worst))
    if len(sys.argv) > 2:
        B, _ = load(sys.argv[2]); summary("B", B)
        common = sorted(set(A) & set(B))
        own_b = gains(B, layers=common); xfer = gains(A, B, layers=common)
        for i, b in enumerate(BUDGETS):
            print(f"  budget {b:>2}: B oracle {st.mean(r[2][i] for r in own_b):.3f}   "
                  f"placed from A, scored on B {st.mean(r[2][i] for r in xfer):.3f}")


if __name__ == "__main__":
    main()

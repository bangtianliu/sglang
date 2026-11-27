"""Second-copy placement for the table-driven 2-copy expert assignment (plain Python, no GPU).

place(freq, budget) -> masks[a][k], a = home rank 0..3, k = 0..2: uint64 bitmaps over local
expert ids (0..63) of home rank a whose copy B lives on rank (a + 1 + k) % 4.

Rule: replicate the `budget` most frequent experts. In descending frequency, each copy goes to
the eligible rank (not its home, under `cap` hosted copies) with the least frequency mass
received so far, ties broken round-robin per home rank. With uniform frequencies this reduces
to round-robin over the other three ranks; with hot experts clustered on one home it spreads
their copies over all three others. `cap` bounds B-weight memory per rank (default even split).
"""
import math

E, R, LOCAL = 256, 4, 64


def place(freq, budget, cap=None, hosted=None):
    """hosted: per-rank copy counts carried across layers (updated in place); cap then bounds
    the running total, so a whole model stays within a per-rank memory budget."""
    assert len(freq) == E and 0 <= budget <= E
    shared = hosted is not None
    hosted = hosted if shared else [0] * R
    cap = (math.ceil(budget / R) if cap is None else cap) if not shared else cap
    # equal frequencies alternate home ranks, so a uniform budget is spread over all homes
    order = sorted(range(E), key=lambda e: (-freq[e], e % LOCAL, e // LOCAL))[:budget]
    masks = [[0, 0, 0] for _ in range(R)]
    mass = [0.0] * R
    turn = [0] * R
    for e in order:
        a, local = divmod(e, LOCAL)
        cands = [k for k in range(3) if cap is None or hosted[(a + 1 + k) % R] < cap]
        if not cands:
            if shared:  # hard memory cap: this expert stays single-homed
                continue
            cands = [min(range(3), key=lambda k: hosted[(a + 1 + k) % R])]  # soft per-layer cap
        start = turn[a] % 3
        k = min(cands, key=lambda k: (mass[(a + 1 + k) % R], (k - start) % 3))
        turn[a] += 1
        b = (a + 1 + k) % R
        masks[a][k] |= 1 << local
        hosted[b] += 1
        mass[b] += freq[e]
    return masks


EXPERT_GIB = 26.7e6 / 2**30  # one MXFP4 expert (gate, up, down + scales), from config.json


def place_model(freqs, gib_per_rank, expert_gib=EXPERT_GIB, grid=(0, 4, 8, 16, 32, 64, 128),
                steps=150, seed=0):
    """Per-layer masks for a whole model under a hard per-rank memory cap for second copies.

    Each layer's busiest-rank load is simulated at the grid budgets (routing sampled from that
    layer's own freqs, 8 rows x top-8); copies then go to the layer segment with the largest
    load reduction per copy until the cap is spent. Returns (budgets, masks, hosted per rank)."""
    from eplb_placement_study import min_max, sample_step
    import random

    per_rank = int(gib_per_rank / expert_gib)
    curves = []
    for l, fl in enumerate(freqs):
        t = float(sum(fl)) or 1.0
        pop = [f / t for f in fl]
        rng = random.Random(seed + l)
        draws = [sorted(set().union(*sample_step(rng, pop, 8, 0))) for _ in range(steps)]
        curve = []
        for b in grid:
            own = owners_from_masks(place(fl, b))
            curve.append(sum(min_max(a, own) for a in draws) / steps)
        curves.append(curve)
    level = [0] * len(freqs)  # index into grid per layer
    left = per_rank * R
    while True:
        best = None
        for l, c in enumerate(curves):
            i = level[l]
            if i + 1 < len(grid) and grid[i + 1] - grid[i] <= left:
                gain = (c[i] - c[i + 1]) / (grid[i + 1] - grid[i])
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, l)
        if best is None:
            break
        l = best[1]
        left -= grid[level[l] + 1] - grid[level[l]]
        level[l] += 1
    budgets = [grid[i] for i in level]
    hosted = [0] * R
    order = sorted(range(len(freqs)), key=lambda l: -budgets[l])
    masks = [None] * len(freqs)
    for l in order:
        masks[l] = place(freqs[l], budgets[l], cap=per_rank, hosted=hosted)
    return budgets, masks, hosted


def owners_from_masks(masks):
    """expert id -> tuple of ranks holding it (home first)."""
    own = {e: (e // LOCAL,) for e in range(E)}
    for a in range(R):
        for k in range(3):
            m = masks[a][k]
            while m:
                low = m & -m
                local = low.bit_length() - 1
                own[a * LOCAL + local] = (a, (a + 1 + k) % R)
                m ^= low
    return own


if __name__ == "__main__":
    # Self-check against the placement study's exact min-max busiest-rank load.
    import random
    import statistics as st
    import sys

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from eplb_placement_study import min_max, popularity, sample_step

    for kind in ("uniform", "zipf=1.0", "zipfidx=1.0"):
        rng = random.Random(3)
        pop = popularity(kind, rng)
        for budget in (0, 32, 64, 128, 256):
            masks = place(pop, budget)
            own = owners_from_masks(masks)
            assert sum(len(v) == 2 for v in own.values()) == budget or budget > 0
            hosted = [0] * R
            for v in own.values():
                if len(v) == 2:
                    hosted[v[1]] += 1
            srng = random.Random(9)
            busy, ideal = [], []
            for _ in range(400):
                active = sorted(set().union(*sample_step(srng, pop, 8, 0)))
                busy.append(min_max(active, own))
                ideal.append(math.ceil(len(active) / R))
            print(f"{kind:<12} budget {budget:>3}  busiest/ideal {st.mean(busy)/st.mean(ideal):.3f}"
                  f"  hosted copies per rank {hosted}")

# Whole-model check: global allocation (place_model) vs the same total split evenly per layer.
import sys, random, math, statistics as st
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else __file__.rsplit("/", 1)[0])
from eplb_place import place, place_model, owners_from_masks, R, EXPERT_GIB
from eplb_placement_study import min_max, sample_step
L, STEPS = 12, 300
rng = random.Random(5)
freqs = []
for l in range(L):  # layers differ: flat to very concentrated, hot sets scattered or clustered
    s = [0.0, 0.6, 1.0, 1.4][l % 4]
    w = [1 / (i + 1) ** s for i in range(256)]
    if l % 3: rng.shuffle(w)
    freqs.append([x * 1e6 for x in w])
def model_busy(masks_per_layer):
    tot = 0.0
    for l, m in enumerate(masks_per_layer):
        own = owners_from_masks(m); srng = random.Random(100 + l); pop = freqs[l]
        tot += st.mean(min_max(sorted(set().union(*sample_step(srng, pop, 8, 0))), own) for _ in range(STEPS))
    return tot
base = model_busy([place(f, 0) for f in freqs])
for gib in (0.2, 0.4, 0.8, 1.6):
    budgets, masks, _ = place_model(freqs, gib)
    total = sum(budgets); even = total // L // R * R
    hosted = [0] * R
    for m in masks:
        for v in owners_from_masks(m).values():
            if len(v) == 2: hosted[v[1]] += 1
    g = model_busy(masks); e = model_busy([place(f, even) for f in freqs])
    print(f"cap {gib:.1f} GiB/rank/{L} layers: copies {total:>4} budgets {budgets}  hosted/rank {hosted} "
          f"(max {max(hosted)*EXPERT_GIB:.2f} GiB)  sum busiest: EP4 {base:.1f}  even {e:.1f}  global {g:.1f}")

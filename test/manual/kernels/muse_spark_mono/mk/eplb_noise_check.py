# How much does placement quality depend on routing-count quality? Masks are built from counts
# of N observed tokens (each top-8 from the true popularity), then scored on the TRUE routing.
# Score = mean busiest-rank active experts (8 rows x top-8), relative to the EP4 baseline.
import sys, random, statistics as st
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else __file__.rsplit("/", 1)[0])
from eplb_place import place, owners_from_masks
from eplb_placement_study import min_max, sample_step, popularity
STEPS = 400
def score(pop, masks, seed):
    own = owners_from_masks(masks); r = random.Random(seed)
    return st.mean(min_max(sorted(set().union(*sample_step(r, pop, 8, 0))), own) for _ in range(STEPS))
def counts(pop, ntok, rng):
    c = [0] * 256
    for _ in range(ntok):
        for e in sample_step(rng, pop, 1, 0)[0]:
            c[e] += 1
    return c
for kind in ("zipf=0.6", "zipf=1.0", "zipfidx=1.0"):
    pop = popularity(kind, random.Random(3))
    base = score(pop, place(pop, 0), 11)
    print(f"\n## {kind}  EP4 busiest {base:.2f}")
    for budget in (8, 16, 32):
        row = [f"oracle {score(pop, place(pop, budget), 11) / base:.3f}"]
        for ntok in (64, 512, 4096, 32768):
            vals = []
            for rep in range(3):
                c = counts(pop, ntok, random.Random(1000 * rep + ntok))
                vals.append(score(pop, place(c, budget), 11) / base)
            row.append(f"N={ntok}: {st.mean(vals):.3f}")
        print(f"  budget {budget:>2}  " + "  ".join(row))

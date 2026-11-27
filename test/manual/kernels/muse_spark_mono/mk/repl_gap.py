"""MK.94 (CPU only): how far is the kernel's levelling rule from the exact optimum?

For placement tables from eplb_place.place() at several budgets and three popularity models,
compares the busiest rank's distinct routed experts under EP4, under the kernel rule
(tlx_route.repl_local, restated here in plain Python) and under the exact min-max assignment
(bipartite matching), against the ideal ceil(active / 4)."""
import math
import random
import statistics as st
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from eplb_place import owners_from_masks, place  # noqa: E402
from eplb_placement_study import min_max, popularity, sample_step  # noqa: E402


def rule_max(active, masks, passes):
    g = [[0] * 3 for _ in range(4)]
    load = [0] * 4
    for e in active:
        a, ep = e >> 6, e & 63
        load[a] += 1
        for k in range(3):
            if masks[a][k] >> ep & 1:
                g[a][k] += 1
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
    return max(load)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    print(f"{'popularity':<13}{'budget':>7}{'EP4':>8}{'rule x2':>9}{'rule x3':>9}{'rule x4':>9}"
          f"{'exact':>8}{'ideal':>8}{'x3 > exact':>12}{'worst gap':>10}")
    for kind in ("uniform", "zipf=1.0", "zipfidx=1.0"):
        pop = popularity(kind, random.Random(3))
        for budget in (0, 32, 64, 128, 256):
            masks = place(pop, budget)
            own = owners_from_masks(masks)
            rng = random.Random(9)
            rows = []
            for _ in range(steps):
                active = sorted(set().union(*sample_step(rng, pop, 8, 0)))
                ep4 = max(sum(1 for e in active if e >> 6 == r) for r in range(4))
                r1, r2, r3 = (rule_max(active, masks, p) for p in (2, 3, 4))
                rows.append((ep4, r1, r2, r3, min_max(active, own), math.ceil(len(active) / 4)))
            mean = [st.mean(x[i] for x in rows) for i in range(6)]
            worse = sum(x[2] > x[4] for x in rows) / steps
            gap = max(x[2] - x[4] for x in rows)
            print(f"{kind:<13}{budget:>7}" + "".join(f"{v:>{w}.2f}" for v, w in zip(mean, (8, 9, 9, 9, 8, 8)))
                  + f"{worse:>11.1%}{gap:>10}")


if __name__ == "__main__":
    main()

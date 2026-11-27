# Offline placement study for 2-copy expert replication (EP4, 256 experts, top-8).
# Load metric: distinct active experts per rank (= weight blocks streamed at small batch),
# plus slot routes per rank. Min-max assignment solved exactly (bipartite matching).
import random, statistics as st, sys, math

E, R, K = 256, 4, 8
import os
FRACS = tuple(float(x) for x in os.environ.get("FRACS", "0,0.25,0.5,1").split(","))
HOME = [e // 64 for e in range(E)]


def popularity(kind, rng):
    if kind == "uniform":
        w = [1.0] * E
    else:
        s = float(kind.split("=")[1])
        w = [1 / (i + 1) ** s for i in range(E)]
        if not kind.startswith("zipfidx"):
            rng.shuffle(w)  # hot experts scattered over ids (zipfidx: hot = low ids = rank 0)
    t = sum(w)
    return [x / t for x in w]


def sample_step(rng, pop, rows, cofire):
    """rows x top-K distinct experts. cofire>0: a per-step 'topic' of 16 experts gets its
    weight multiplied by (1+cofire), so rows of one step share experts."""
    w = pop[:]
    if cofire:
        for e in rng.sample(range(E), 16):
            w[e] *= 1 + cofire
    out = []
    for _ in range(rows):
        picks = set()
        while len(picks) < K:
            picks.add(rng.choices(range(E), weights=w)[0])
        out.append(picks)
    return out


def min_max(active, owners):
    """Exact min over assignments of max per-rank count (each active expert on one owner)."""
    lo = math.ceil(len(active) / R)
    fixed = [0] * R
    flex = []
    for e in active:
        if len(owners[e]) == 1:
            fixed[owners[e][0]] += 1
        else:
            flex.append(e)
    lo = max(lo, max(fixed))
    for cap in range(lo, len(active) + 1):
        room = [cap - f for f in fixed]
        match = {}

        def aug(e, seen):
            for r in owners[e]:
                if r in seen:
                    continue
                seen.add(r)
                if room[r] > 0:
                    room[r] -= 1
                    match[e] = r
                    return True
                for e2 in [x for x, rr in match.items() if rr == r]:
                    if aug(e2, seen):
                        match[e] = r
                        return True
            return False

        if all(aug(e, set()) for e in flex):
            return cap
    return len(active)


def tables(pop, frac, rng, est_noise):
    """Second-home tables. Replicate the top `frac` of experts by (noisy) estimated popularity."""
    est = [p * math.exp(rng.gauss(0, est_noise)) for p in pop]
    order = sorted(range(E), key=lambda e: -est[e])
    rep = set(order[: int(round(frac * E))])
    t = {}
    # neighbour: next rank
    t["neighbour"] = {e: ((HOME[e], (HOME[e] + 1) % R) if e in rep else (HOME[e],)) for e in range(E)}
    # random other rank
    t["random"] = {e: ((HOME[e], rng.choice([r for r in range(R) if r != HOME[e]])) if e in rep
                       else (HOME[e],)) for e in range(E)}
    # balanced design: replicated experts of each rank spread round-robin over the other 3 ranks
    bal = {}
    cnt = [0] * R
    for e in order:
        if e in rep:
            others = [r for r in range(R) if r != HOME[e]]
            bal[e] = (HOME[e], others[cnt[HOME[e]] % 3])
            cnt[HOME[e]] += 1
        else:
            bal[e] = (HOME[e],)
    t["balanced"] = bal
    # popularity-aware: hottest first, second copy on the rank with least estimated hot mass
    mass = [sum(est[e] for e in range(E) if HOME[e] == r) for r in range(R)]
    pa = {}
    for e in order:
        if e in rep:
            r2 = min((r for r in range(R) if r != HOME[e]), key=lambda r: mass[r])
            mass[r2] += est[e] / 2
            mass[HOME[e]] -= est[e] / 2
            pa[e] = (HOME[e], r2)
        else:
            pa[e] = (HOME[e],)
    t["pop-aware"] = pa
    return t


def study(kind, cofire, rows, trials, seed=1, est_noise=0.3):
    rng = random.Random(seed)
    pop = popularity(kind, rng)
    res = {}
    for frac in FRACS:
        tbl = tables(pop, frac, random.Random(seed + 7), est_noise)
        names = ["EP4"] if frac == 0 else list(tbl)
        for name in names:
            owners = {e: (HOME[e],) for e in range(E)} if frac == 0 else tbl[name]
            srng = random.Random(seed + 99)  # same draws for every table
            busy, ideal = [], []
            for _ in range(trials):
                step = sample_step(srng, pop, rows, cofire)
                active = sorted(set().union(*step))
                busy.append(min_max(active, owners))
                ideal.append(math.ceil(len(active) / R))
            busy.sort()
            res[(frac, name)] = (st.mean(busy), busy[int(0.9 * len(busy))], st.mean(ideal))
    return res


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    for kind, cofire, rows in [("uniform", 0, 8), ("zipf=0.8", 0, 8), ("zipf=1.2", 0, 8),
                               ("zipfidx=1.0", 0, 8), ("zipf=1.0", 20, 8), ("zipf=1.0", 0, 32)]:
        r = study(kind, cofire, rows, trials)
        ideal = next(iter(r.values()))[2]
        print(f"\n## {kind} cofire={cofire} rows={rows}   ideal busiest = {ideal:.2f}")
        for (frac, name), (m, p90, _) in r.items():
            print(f"  {frac:>4.2f}x extra  {name:<10} mean {m:6.2f}  p90 {p90:3d}  vs ideal {m/ideal:5.3f}")

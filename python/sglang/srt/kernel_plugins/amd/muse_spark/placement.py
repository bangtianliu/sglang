"""Second-copy placement for the gfx1250 chain's replicated experts (plain Python, no GPU).

``place(freq, budget)`` -> ``masks[a][k]`` (a = home rank 0..3, k = 0..2): uint64 bitmaps over
local expert ids of home rank a whose second copy lives on rank ``(a + 1 + k) % 4``; the format
``tlx_route`` takes. Written by the h13-17 session (mk/eplb_place.py); kept identical here.

Rule: replicate the ``budget`` most frequent experts. In descending frequency, each copy goes
to the eligible rank (not its home, under ``cap`` hosted copies) with the least frequency mass
received so far, ties broken round-robin per home rank. ``cap`` bounds the second-copy memory
per rank (default: an even split).
"""

import math

E, R, LOCAL = 256, 4, 64


def place(freq, budget, cap=None):
    assert len(freq) == E and 0 <= budget <= E
    cap = math.ceil(budget / R) if cap is None else cap
    assert cap * R >= budget, "cap too small for budget"
    # equal frequencies alternate home ranks, so a uniform budget is spread over all homes
    order = sorted(range(E), key=lambda e: (-freq[e], e % LOCAL, e // LOCAL))[:budget]
    masks = [[0, 0, 0] for _ in range(R)]
    hosted = [0] * R
    mass = [0.0] * R
    turn = [0] * R
    for e in order:
        a, local = divmod(e, LOCAL)
        cands = [k for k in range(3) if hosted[(a + 1 + k) % R] < cap]
        if not cands:  # every other rank is at cap: overflow to the least-loaded one
            cands = [min(range(3), key=lambda k: hosted[(a + 1 + k) % R])]
        start = turn[a] % 3
        k = min(cands, key=lambda k: (mass[(a + 1 + k) % R], (k - start) % 3))
        turn[a] += 1
        b = (a + 1 + k) % R
        masks[a][k] |= 1 << local
        hosted[b] += 1
        mass[b] += freq[e]
    return masks


def bits(mask):
    """Set bit positions of ``mask``, ascending."""
    return [i for i in range(64) if mask >> i & 1]


def reference_locals(ids, masks, rank, passes=3):
    """The assignment rule of ``tlx_route.repl_local`` in plain Python, for one rank: which of
    the 64 routes (8 rows x 8 global expert ids) this rank runs (``mine``) and the local expert
    index of every distinct expert it runs (``locals``, 0..63 own, 64.. compact second copies)."""
    present = sorted(e for e in set(ids) if 0 <= e < E)  # other ids are not routes

    def grp(e):
        a, ep = e >> 6, e & 63
        for k in range(3):
            if masks[a][k] >> ep & 1:
                return k
        return 3

    g = [[0] * 3 for _ in range(R)]
    load = [0] * R
    for e in present:
        load[e >> 6] += 1
        if grp(e) < 3:
            g[e >> 6][grp(e)] += 1
    m = [[0] * 3 for _ in range(R)]
    for _ in range(passes):
        for a in range(R):
            for k in range(3):
                t = (a + 1 + k) & 3
                d = load[a] - load[t]
                h = d >> 1 if d >= 0 else -((-d) >> 1)
                mv = max(min(g[a][k] - m[a][k], h), -m[a][k])
                m[a][k] += mv
                load[a] -= mv
                load[t] += mv
    base = []
    for t in range(R):
        n0 = bin(masks[(t - 1) % R][0]).count("1")
        n1 = bin(masks[(t - 2) % R][1]).count("1")
        base.append([0, n0, n0 + n1])
    place_of = {}
    seen = [[0] * 3 for _ in range(R)]
    for e in present:
        a, ep = e >> 6, e & 63
        k = grp(e)
        moved = k < 3 and seen[a][k] < m[a][k]
        if k < 3:
            seen[a][k] += 1
        if moved:
            t = (a + 1 + k) & 3
            place_of[e] = (t, 64 + base[t][k] + bin(masks[a][k] & ((1 << ep) - 1)).count("1"))
        else:
            place_of[e] = (a, ep)
    return dict(mine=[1 if e in place_of and place_of[e][0] == rank else 0 for e in ids],
                locals=[place_of[e][1] for e in present if place_of[e][0] == rank])

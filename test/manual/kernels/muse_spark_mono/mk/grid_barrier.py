"""GPU0: is one kernel per layer cheaper than a chain? 11 in-kernel grid barriers
(MK.0 cheap-handoff recipe: SCOPE_DEV store + s_wait_storecnt, relaxed atomics on 32
replicated counters, 1-lane bounded SCOPE_DEV poll, first-touch cached read of another
CTA's word) vs a graph of 12 kernels doing the same per-phase write/read. Same grid."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from handoff_floor import _clock, _drain0, _ld32, _st32  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

PHASES = 12


@triton.jit
def persistent(Data, Cnt, Ep, Fail, G: tl.constexpr, NB: tl.constexpr, SLEEP: tl.constexpr = 0,
               Stamps=None, DST: tl.constexpr = 0):
    pid = tl.program_id(0)
    lane = tl.arange(0, 32)
    epoch = tl.load(Ep + pid) + 1
    tl.store(Ep + pid, epoch)
    bad = 0
    for k in tl.static_range(NB + 1):
        # phase k: write my word
        val = epoch * 1000 + k * 7 + pid
        t0 = _clock()
        if DST == 0:
            z = _st32(Data + k * G + pid + lane * 0, val + lane * 0)
        elif DST >= 3:
            tl.atomic_xchg(Data + k * G + pid + lane * 0, val + lane * 0, mask=lane == 0,
                           sem="relaxed", scope="gpu")
            z = lane * 0
        elif DST == 1:
            tl.store(Data + k * G + pid + lane * 0, val + lane * 0)
            z = lane * 0
        else:
            z = lane * 0
        if k < NB:
            zz = _drain0(z)
            tl.debug_barrier()
            t1 = _clock()
            # arrival: one returning atomic per CTA; the last arriver releases 32 flag lines;
            # every CTA polls its own flag line (G / 32 pollers per line)
            # one thread arrives and one thread polls (gfx950 mailbox rule: throttled polling)
            t = tl.arange(0, 128)
            one = t == 0
            if DST == 4:
                # tree: 32 group counters, each group's last arriver bumps the top counter
                old = tl.atomic_add(Cnt + k * 4096 + 1056 + (pid % 32) * 32 + tl.max(zz, 0) + t * 0, 1,
                                    mask=one, sem="relaxed", scope="gpu")
                glast = tl.max(tl.where(one, old, -1), 0) == (G // 32) * epoch - 1
                last = glast & False
                if glast:
                    top = tl.atomic_add(Cnt + k * 4096 + t * 0, 1, mask=one, sem="relaxed", scope="gpu")
                    last = tl.max(tl.where(one, top, -1), 0) == 32 * epoch - 1
            else:
                old = tl.atomic_add(Cnt + k * 4096 + tl.max(zz, 0) + t * 0, 1, mask=one,
                                    sem="relaxed", scope="gpu")
                last = tl.max(tl.where(one, old, -1), 0) == G * epoch - 1
            t2 = _clock()
            if last:
                if DST >= 3:
                    tl.atomic_xchg(Cnt + k * 4096 + 32 + lane * 32, epoch + lane * 0, sem="relaxed",
                                   scope="gpu")
                else:
                    _st32(Cnt + k * 4096 + 32 + lane * 32, epoch + lane * 0)
            target = epoch
            flag = Cnt + k * 4096 + 32 + (pid % 32) * 32 + t * 0
            c = tl.max(tl.where(one, tl.atomic_add(flag, 0, mask=one, sem="relaxed", scope="gpu"), -1), 0)
            n = 0
            while (c < target) & (n < 20000):
                if SLEEP:
                    tl.inline_asm_elementwise("s_sleep 1\nv_mov_b32 $0, 0", "=v", [], dtype=tl.int32,
                                              is_pure=False, pack=1)
                c = tl.max(tl.where(one, tl.atomic_add(flag, 0, mask=one, sem="relaxed", scope="gpu"), -1), 0)
                n += 1
            dep = tl.minimum(c - target, 0)
            t3 = _clock()
            if Stamps is not None:
                q = tl.arange(0, 4)
                tl.store(Stamps + (pid * 16 + k) * 4 + q,
                         tl.where(q == 0, t0, tl.where(q == 1, t1, tl.where(q == 2, t2, t3))).to(tl.int64))
            tl.debug_barrier()
            other = (pid + 1 + k) % G
            if DST >= 3:
                got = tl.max(tl.where(one, tl.atomic_add(Data + dep + k * G + other + t * 0, 0, mask=one,
                                                         sem="relaxed", scope="gpu"), 0), 0)
            else:
                got = tl.max(tl.load(Data + dep + k * G + other + lane * 0), 0)
            if DST == 0 or DST >= 3:
                bad += (got != epoch * 1000 + k * 7 + other).to(tl.int32)
    tl.store(Fail + pid, tl.load(Fail + pid) + bad)


@triton.jit
def phase_kernel(Data, Ep, Fail, k, G: tl.constexpr, FIRST: tl.constexpr):
    pid = tl.program_id(0)
    if FIRST:
        epoch = tl.load(Ep + pid) + 1
        tl.store(Ep + pid, epoch)
    else:
        epoch = tl.load(Ep + pid)
    if k > 0:
        other = (pid + k) % G
        got = tl.load(Data + (k - 1) * G + other)
        tl.store(Fail + pid, tl.load(Fail + pid) + (got != epoch * 1000 + (k - 1) * 7 + other).to(tl.int32))
    tl.store(Data + k * G + pid, epoch * 1000 + k * 7 + pid)


def timed(g, n=60):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return round(sorted(ts)[n // 2], 2)


def main():
    require_gpu0()
    keep = []
    for G in (192, 384, 576):
        data = torch.zeros(PHASES * G, device="cuda", dtype=torch.int32)
        cnt = torch.zeros(PHASES * 4096, device="cuda", dtype=torch.int32)
        ep = torch.zeros(G, device="cuda", dtype=torch.int32)
        ep2 = torch.zeros(G, device="cuda", dtype=torch.int32)
        fail = torch.zeros(G, device="cuda", dtype=torch.int32)
        row = dict(grid=G)
        for nb, sl, dv in ((0, 0, 0), (PHASES - 1, 0, 0), (PHASES - 1, 0, 3), (PHASES - 1, 0, 4)):
            ep = torch.zeros(G, device="cuda", dtype=torch.int32)  # epochs count this variant's launches
            keep.append(ep)
            cnt.zero_()
            persistent[(G,)](data, cnt, ep, fail, G, nb, sl, None, dv, num_warps=4)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                persistent[(G,)](data, cnt, ep, fail, G, nb, sl, None, dv, num_warps=4)
            keep.append(g)
            row[f"one_kernel_{nb}_barriers{['', '', '', '_atomic', '_tree'][dv]}_us"] = timed(g)
        for k in range(PHASES):
            phase_kernel[(G,)](data, ep2, fail, k, G, k == 0, num_warps=4)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for k in range(PHASES):
                phase_kernel[(G,)](data, ep2, fail, k, G, k == 0, num_warps=4)
        keep.append(g)
        row["chain_12_kernels_us"] = timed(g)
        torch.cuda.synchronize()
        for dst in (3, 4):
            st = torch.zeros(G * 16 * 4, device="cuda", dtype=torch.int64)
            ep3 = torch.zeros(G, device="cuda", dtype=torch.int32)
            cnt.zero_()
            for _ in range(3):
                persistent[(G,)](data, cnt, ep3, fail, G, PHASES - 1, 0, st, dst, num_warps=4)
            torch.cuda.synchronize()
            v = st.view(G, 16, 4)[:, : PHASES - 1].double() * 0.01  # 100 MHz ticks -> us
            med = lambda x: round(float(x.median()), 2)
            arr = v[:, :, 2]
            row[f"stamps_dst{dst}"] = dict(
                store_drain=med(v[:, :, 1] - v[:, :, 0]), arrive_atomic=med(v[:, :, 2] - v[:, :, 1]),
                wait_flag=med(v[:, :, 3] - v[:, :, 2]),
                arrival_spread=med(arr.max(0).values - arr.min(0).values))
        row["failures"] = int(fail.sum())
        for v in ("", "_atomic", "_tree"):
            row[f"per_barrier{v}_us"] = round((row[f"one_kernel_{PHASES-1}_barriers{v}_us"]
                                               - row["one_kernel_0_barriers_us"]) / (PHASES - 1), 2)
        row["per_boundary_us"] = round(row["chain_12_kernels_us"] / PHASES, 2)
        print(json.dumps(row), flush=True)
        keep.append((data, cnt, ep, ep2, fail))


if __name__ == "__main__":
    main()

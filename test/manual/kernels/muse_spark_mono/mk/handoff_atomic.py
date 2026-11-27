"""GPU0: bulk in-kernel handoff on the atomic path (MK.6 shape: 8 producers publish 8 x 4096
int32 words, 72 consumers wait and checksum all of it). Times from the first producer start.
MODE 0: SCOPE_DEV stores + s_wait_storecnt + 32 replicated counters (MK.0/MK.6 recipe), plain reads
MODE 3: atomic_xchg data (returns consumed) + atomic counter; 1-thread atomic poll; plain reads
MODE 4: as 3, consumers read with atomic loads
MODE 5: tagged pairs (tag<<32|word) by atomic_xchg int64, no counter; consumers atomic-load pairs,
        re-load stale ones (bounded)"""
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

ROWS, WORDS, NCONS = 8, 4096, 72


@triton.jit
def kernel(Data, Pairs, Cnt, Ep, Stamps, Out, Fail, MODE: tl.constexpr, ROWS: tl.constexpr,
           WORDS: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, 128)
    one = lane == 0
    epoch = tl.load(Ep + pid) + 1
    tl.store(Ep + pid, epoch)
    PER: tl.constexpr = WORDS // 128
    if pid < ROWS:
        for _ in tl.static_range(4):
            tl.inline_asm_elementwise("s_sleep 127\nv_mov_b32 $0, 0", "=v", [], dtype=tl.int32,
                                      is_pure=False, pack=1)
        ts = _clock()
        dep = lane * 0
        for k in tl.static_range(PER):
            w = pid * WORDS + k * 128 + lane
            val = epoch * 7 + w
            if MODE == 0:
                dep += _st32(Data + w, val)
            elif MODE == 5:
                pair = (epoch.to(tl.int64) << 32) | (val.to(tl.int64) & 0xFFFFFFFF)
                dep += tl.atomic_xchg(Pairs + w, pair, sem="relaxed", scope="gpu").to(tl.int32) * 0
            else:
                dep += tl.atomic_xchg(Data + w, val, sem="relaxed", scope="gpu") * 0
        if MODE == 0:
            dep = _drain0(dep)
        if MODE != 5:
            tl.debug_barrier()
            tl.atomic_add(Cnt + lane * 32 + tl.max(dep, 0), 1, mask=lane < 32, sem="relaxed",
                          scope="gpu")
        tp = _clock()
        q = tl.arange(0, 4)
        tl.store(Stamps + pid * 4 + q, tl.where(q == 0, ts, tl.where(q == 1, tp, 0)).to(tl.int64))
    else:
        c = pid - ROWS
        acc = lane * 0
        n = 0
        if MODE != 5:
            target = ROWS * epoch
            ctr = Cnt + (c % 32) * 32 + lane * 0
            if MODE == 0:
                v = tl.min(_ld32(ctr), 0)
            else:
                v = tl.max(tl.where(one, tl.atomic_add(ctr, 0, mask=one, sem="relaxed", scope="gpu"), 0), 0)
            while (v < target) & (n < 100000):
                if MODE == 0:
                    v = tl.min(_ld32(ctr), 0)
                else:
                    v = tl.max(tl.where(one, tl.atomic_add(ctr, 0, mask=one, sem="relaxed", scope="gpu"), 0), 0)
                n += 1
            d = tl.minimum(v - target, 0)
            for r in tl.static_range(ROWS):
                for k in tl.static_range(PER):
                    dp = Data + d + r * WORDS + k * 128 + lane
                    if MODE == 4:
                        acc += tl.atomic_add(dp, 0, sem="relaxed", scope="gpu")
                    else:
                        acc += tl.load(dp)
        else:
            for r in tl.static_range(ROWS):
                for k in tl.static_range(PER):
                    pp = Pairs + r * WORDS + k * 128 + lane
                    p = tl.atomic_add(pp, 0, sem="relaxed", scope="gpu")
                    m = 0
                    while (tl.min(((p >> 32).to(tl.int32) == epoch).to(tl.int32), 0) == 0) & (m < 100000):
                        p = tl.atomic_add(pp, 0, sem="relaxed", scope="gpu")
                        m += 1
                    n += m
                    acc += (p & 0xFFFFFFFF).to(tl.int32)
        td = _clock()
        total = tl.sum(acc, 0)
        want = 7 * epoch * ROWS * WORDS + (ROWS * WORDS // 2) * (ROWS * WORDS - 1)
        tl.store(Fail + c, tl.load(Fail + c) + (total != want).to(tl.int32))
        q = tl.arange(0, 4)
        tl.store(Stamps + pid * 4 + q, tl.where(q == 2, td, tl.where(q == 3, n, 0)).to(tl.int64))


def main():
    require_gpu0()
    G = ROWS + NCONS
    for mode in (0, 3, 4, 5):
        data = torch.zeros(ROWS * WORDS, device="cuda", dtype=torch.int32)
        pairs = torch.zeros(ROWS * WORDS, device="cuda", dtype=torch.int64)
        cnt = torch.zeros(1024, device="cuda", dtype=torch.int32)
        ep = torch.zeros(G, device="cuda", dtype=torch.int32)
        st = torch.zeros(G * 4, device="cuda", dtype=torch.int64)
        out = torch.zeros(G, device="cuda", dtype=torch.int32)
        fail = torch.zeros(NCONS, device="cuda", dtype=torch.int32)
        pub, med, last = [], [], []
        for _ in range(20):
            kernel[(G,)](data, pairs, cnt, ep, st, out, fail, mode, ROWS, WORDS, num_warps=4)
            torch.cuda.synchronize()
            v = st.view(G, 4).cpu().double()
            t0 = v[:ROWS, 0].min()
            pub.append(float((v[:ROWS, 1] - t0).max()) * 0.01)
            done = (v[ROWS:, 2] - t0) * 0.01
            med.append(float(done.median())); last.append(float(done.max()))
        m = lambda x: round(sorted(x)[len(x) // 2], 2)
        print(json.dumps(dict(mode=mode, published_us=m(pub), consumers_done_med_us=m(med),
                              consumers_done_last_us=m(last), failures=int(fail.sum()))), flush=True)


if __name__ == "__main__":
    main()

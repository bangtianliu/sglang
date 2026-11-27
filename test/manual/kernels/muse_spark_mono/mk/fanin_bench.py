"""MK.0: cost of an in-kernel small fan-in handoff vs a kernel boundary (GPU0 only).

8 producer CTAs each write one 4096-wide BF16 row (2048 words) R times (replicas),
publish a per-row flag on its own 128 B line. C consumer CTAs wait for all 8 flags,
then read all 8 rows (64 KiB) from replica (pid % R) and write a checksum.

MODE 0: consumer reads with .cv (scope SYS, bypasses caches)
MODE 1: consumer issues one global_inv SCOPE_DEV, then default cached loads
BOUNDARY: producer kernel, then consumer kernel with default loads (graph boundary)
"""

import json
import os
import sys

import torch
import triton
import triton.language as tl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_memory import load_flag

assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
WORDS = 2048  # one row of 4096 bf16
ROWS = 8


@triton.jit
def _st_dev(ptr, v):
    return tl.inline_asm_elementwise(
        "global_store_b32 $1, $2, off scope:SCOPE_DEV\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        constraints="=&v,v,v",
        args=[ptr, v],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _drain(x):
    # release: write back + drain stores; result is 0, used to order later ops
    return tl.inline_asm_elementwise(
        "global_wb scope:SCOPE_DEV\ns_wait_storecnt 0x0\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        constraints="=v,v",
        args=[x],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _drain_nowb(x):
    # drain own SCOPE_DEV stores only (no cache writeback)
    return tl.inline_asm_elementwise(
        "s_wait_storecnt 0x0\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        constraints="=v,v",
        args=[x],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ld_dev(ptr):
    return tl.inline_asm_elementwise(
        "global_load_b32 $0, $1, off scope:SCOPE_DEV\ns_wait_loadcnt 0x0\ns_wait_xcnt 0x0",
        constraints="=v,v",
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _inv(x):
    return tl.inline_asm_elementwise(
        "global_inv scope:SCOPE_DEV\ns_wait_loadcnt 0x0\nv_mov_b32 $0, 0",
        constraints="=v,v",
        args=[x],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def fused(Act, Cnt, Ep, Out, Fail, R: tl.constexpr, MODE: tl.constexpr,
          WORDS: tl.constexpr, ROWS: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    epoch = tl.load(Ep + pid) + 1
    tl.store(Ep + pid, epoch)
    idx = tl.arange(0, WORDS)
    if pid < ROWS:
        vals = epoch * 131 + pid * WORDS + idx
        z = idx * 0
        if (MODE < 3) | (MODE >= 5):
            for r in tl.static_range(R):
                z += _st_dev(Act + r * ROWS * WORDS + pid * WORDS + idx, vals)
        if MODE == 8:
            one = tl.arange(0, 128) * 0 + tl.max(z, 0)
            z = z + tl.max(_drain(one), 0)
        elif MODE < 4:
            z = _drain(z)
        elif MODE >= 5:
            z = _drain_nowb(z)
        tl.debug_barrier()
        kk = tl.arange(0, K)
        tl.atomic_add(Cnt + kk * 32 + tl.max(z, 0), kk * 0 + 1, sem="relaxed", scope="gpu")
    else:
        target = ROWS * epoch
        pre = tl.zeros([WORDS], tl.int32)
        if MODE == 7:  # deliberate hazard: touch the buffer before the wait
            for row in tl.static_range(ROWS):
                pre += tl.load(Act + (pid % R) * ROWS * WORDS + row * WORDS + idx)
        c = load_flag(Cnt + (pid % K) * 32)
        n = 0
        while (c < target) & (n < 200000):
            c = load_flag(Cnt + (pid % K) * 32)
            n += 1
        ok = c >= target
        dep = tl.minimum(c - target, 0)  # 0; orders reads after the poll
        base = Act + (pid % R) * ROWS * WORDS + dep
        if (MODE == 1) | (MODE == 8):
            base += tl.max(_inv(tl.arange(0, 128) * 0 + dep), 0)
        acc = tl.zeros([WORDS], tl.int32)
        if (MODE < 2) | (MODE >= 5):
            for row in tl.static_range(ROWS):
                if (MODE == 0) | (MODE == 5):
                    acc += tl.load(base + row * WORDS + idx, cache_modifier=".cv")
                else:
                    acc += tl.load(base + row * WORDS + idx)
        total = tl.sum(acc, 0) + dep
        want = 131 * ROWS * WORDS * epoch + (ROWS * WORDS // 2) * (ROWS * WORDS - 1)
        bad = (total != want) & ((MODE < 2) | (MODE >= 5))
        tl.store(Out + pid, total)
        tl.store(Ep + pid, epoch + (tl.sum(pre, 0) == 2147483647).to(tl.int32))
        tl.store(Fail + pid, tl.load(Fail + pid) + tl.where(ok, 0, 1) + tl.where(bad, 1000, 0))


@triton.jit
def produce(Act, Ep, WORDS: tl.constexpr):
    pid = tl.program_id(0)
    epoch = tl.load(Ep + pid) + 1
    tl.store(Ep + pid, epoch)
    idx = tl.arange(0, WORDS)
    tl.store(Act + pid * WORDS + idx, epoch * 131 + pid * WORDS + idx)


@triton.jit
def consume(Act, Ep, Out, WORDS: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    epoch = tl.load(Ep + pid) + 1
    tl.store(Ep + pid, epoch)
    idx = tl.arange(0, WORDS)
    acc = tl.zeros([WORDS], tl.int32)
    for row in tl.static_range(ROWS):
        acc += tl.load(Act + row * WORDS + idx)
    tl.store(Out + pid, tl.sum(acc, 0))


def expected(epoch):
    idx = torch.arange(WORDS, dtype=torch.int64)
    tot = 0
    for p in range(ROWS):
        tot += int((epoch * 131 + p * WORDS + idx).sum())
    return ((tot + 2**31) % 2**32) - 2**31


def bench(fn, iters=200, reps=5):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / iters)
    ts.sort()
    return ts[len(ts) // 2], 1 + reps * iters  # median us/iter, launches done


def main():
    dev = "cuda"
    C = int(sys.argv[1]) if len(sys.argv) > 1 else 184
    res = []
    # boundary baseline
    act = torch.zeros(ROWS * WORDS, dtype=torch.int32, device=dev)
    epp = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    epc = torch.zeros(C, dtype=torch.int32, device=dev)
    out = torch.zeros(C, dtype=torch.int32, device=dev)

    def boundary():
        produce[(ROWS,)](act, epp, WORDS=WORDS, num_warps=4)
        consume[(C,)](act, epc, out, WORDS=WORDS, ROWS=ROWS, num_warps=4)

    us, n = bench(boundary)
    res.append(dict(variant="boundary(2 kernels)", us=us, ok=bool((out == expected(n)).all())))
    print(json.dumps(res[-1]), flush=True)

    def only_consume():
        consume[(C,)](act, epc, out, WORDS=WORDS, ROWS=ROWS, num_warps=4)

    us, _ = bench(only_consume)
    res.append(dict(variant="consumer kernel alone", us=us))
    print(json.dumps(res[-1]), flush=True)

    for mode, R, K in [tuple(map(int, v.split(","))) for v in sys.argv[2].split(";")]:
        if True:
            act = torch.zeros(R * ROWS * WORDS, dtype=torch.int32, device=dev)
            flags = torch.zeros(K * 32, dtype=torch.int32, device=dev)
            ep = torch.zeros(ROWS + C, dtype=torch.int32, device=dev)
            outf = torch.zeros(ROWS + C, dtype=torch.int32, device=dev)
            fail = torch.zeros(ROWS + C, dtype=torch.int32, device=dev)

            def f():
                fused[(ROWS + C,)](act, flags, ep, outf, fail, R=R, MODE=mode, K=K,
                                   WORDS=WORDS, ROWS=ROWS, num_warps=4)

            us, n = bench(f)
            torch.cuda.synchronize()
            good = ((mode >= 2 and mode < 5) or bool((outf[ROWS:] == expected(n)).all())) and int(fail.sum()) == 0
            res.append(dict(variant=f"fused mode={mode} R={R} K={K}", us=us, ok=good,
                            fails=int(fail.sum())))
            print(json.dumps(res[-1]), flush=True)
    print(json.dumps(dict(consumers=C, results=res)))


if __name__ == "__main__" and not os.environ.get("MK_CONCURRENT"):
    main()


@triton.jit
def produce_flag(Act, Cnt, Ep, WORDS: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    epoch = tl.load(Ep + pid) + 1
    tl.store(Ep + pid, epoch)
    idx = tl.arange(0, WORDS)
    z = idx * 0
    z += _st_dev(Act + pid * WORDS + idx, epoch * 131 + pid * WORDS + idx)
    z = _drain_nowb(z)
    tl.debug_barrier()
    kk = tl.arange(0, K)
    tl.atomic_add(Cnt + kk * 32 + tl.max(z, 0), kk * 0 + 1, sem="relaxed", scope="gpu")


@triton.jit
def consume_flag(Act, Cnt, Ep, Out, Fail, WORDS: tl.constexpr, ROWS: tl.constexpr,
                 K: tl.constexpr, SPIN_WORK: tl.constexpr):
    pid = tl.program_id(0)
    epoch = tl.load(Ep + pid) + 1
    tl.store(Ep + pid, epoch)
    target = ROWS * epoch
    c = load_flag(Cnt + (pid % K) * 32)
    n = 0
    while (c < target) & (n < 200000):
        c = load_flag(Cnt + (pid % K) * 32)
        n += 1
    dep = tl.minimum(c - target, 0)
    idx = tl.arange(0, WORDS)
    acc = tl.zeros([WORDS], tl.int32)
    for row in tl.static_range(ROWS):
        if SPIN_WORK == 1:
            acc += tl.load(Act + dep + row * WORDS + idx, cache_modifier=".cv")
        else:
            acc += tl.load(Act + dep + row * WORDS + idx)
    total = tl.sum(acc, 0)
    want = 131 * ROWS * WORDS * epoch + (ROWS * WORDS // 2) * (ROWS * WORDS - 1)
    tl.store(Out + pid, total)
    tl.store(Fail + pid, tl.load(Fail + pid) + tl.where(c >= target, 0, 1) + tl.where(total != want, 1000, 0))


def concurrent(C, K=32):
    act = torch.zeros(ROWS * WORDS, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(K * 32, dtype=torch.int32, device="cuda")
    epp = torch.zeros(ROWS, dtype=torch.int32, device="cuda")
    epc = torch.zeros(C, dtype=torch.int32, device="cuda")
    out = torch.zeros(C, dtype=torch.int32, device="cuda")
    fail = torch.zeros(C, dtype=torch.int32, device="cuda")
    side = torch.cuda.Stream()
    res = []
    for order, cv in (("sequential", 0), ("consumer_first", 1), ("consumer_first", 0)):
        fail.zero_()
        def step():
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            if order == "consumer_first":
                with torch.cuda.stream(side):
                    consume_flag[(C,)](act, cnt, epc, out, fail, WORDS=WORDS, ROWS=ROWS, K=K, SPIN_WORK=cv, num_warps=4)
                produce_flag[(ROWS,)](act, cnt, epp, WORDS=WORDS, K=K, num_warps=4)
            else:
                produce_flag[(ROWS,)](act, cnt, epp, WORDS=WORDS, K=K, num_warps=4)
                consume_flag[(C,)](act, cnt, epc, out, fail, WORDS=WORDS, ROWS=ROWS, K=K, SPIN_WORK=cv, num_warps=4)
            main.wait_stream(side)
        us, _ = bench(step)
        torch.cuda.synchronize()
        e = int(epp[0])
        d = act.view(8, 2048).to(torch.int64) - (e * 131 + torch.arange(16384, device=act.device).view(8, 2048))
        print("ROWDIFF", [int((d[r] != 0).sum()) for r in range(8)], "epp", epp.tolist(), flush=True)
        print("ACT", "hostsum", ((int(act.to(torch.int64).sum()) + 2**31) % 2**32) - 2**31, "out", out[:4].tolist(), act[:3].tolist(), act[2048:2051].tolist(), act[-3:].tolist(), "exp", [e*131, e*131+1, e*131+2], [e*131+2048], flush=True)
        res.append(dict(variant=f"{order} cv={cv}", us=us, fails=int(fail.sum()),
                        epp=epp[:2].tolist(), epc=epc[:2].tolist(), cnt=cnt[0].item(),
                        out0=out[0].item(), want_for_epc=expected(int(epc[0])), want_for_epp=expected(int(epp[0]))))
        print(json.dumps(res[-1]), flush=True)
    return res


if __name__ == "__main__" and os.environ.get("MK_CONCURRENT"):
    concurrent(int(sys.argv[1]))

"""MK.6: handoff latency floor on gfx1250, idle vs under streaming load (GPU0 only).

One launch: 8 producer CTAs publish an x4-sized block (8 rows x 4096 words =
128 KiB of BF16 pairs), 72 consumer CTAs wait for all of it and checksum it,
optional 112 streamer CTAs read a 1 GiB buffer the whole time.
In-kernel realtime stamps: producer start, producer published, consumer done.

MODE 0  counter : SCOPE_DEV b32 asm stores, s_wait_storecnt, relaxed atomic (data
                  dependent), 1-lane poll of a replicated counter, first-touch cached reads
MODE 1  tagged  : one SCOPE_DEV b64 asm store per value (tag<<32 | word); consumers
                  load pairs SCOPE_DEV and re-load the lanes whose tag is stale
MODE 2  tagged x8 replicas: as 1, producers write 8 copies, consumer reads copy pid%8
"""

import json
import os
import sys

import torch
import triton
import triton.language as tl

assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
ROWS, WORDS = 8, 4096  # words per row (BF16 pairs of an 8192-wide row)


@triton.jit
def _clock():
    return tl.inline_asm_elementwise(
        "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\ns_wait_kmcnt 0",
        "=s", [], dtype=tl.uint64, is_pure=False, pack=1)


@triton.jit
def _st32(ptr, v):
    return tl.inline_asm_elementwise(
        "global_store_b32 $1, $2, off scope:SCOPE_DEV\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        "=&v,v,v", [ptr, v], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _st64(ptr, v):
    return tl.inline_asm_elementwise(
        "global_store_b64 $1, $2, off scope:SCOPE_DEV\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        "=&v,v,v", [ptr, v], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ld64(ptr):
    return tl.inline_asm_elementwise(
        "global_load_b64 $0, $1, off scope:SCOPE_DEV\ns_wait_loadcnt 0x0\ns_wait_xcnt 0x0",
        "=v,v", [ptr], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _ld32(ptr):
    return tl.inline_asm_elementwise(
        "global_load_b32 $0, $1, off scope:SCOPE_DEV\ns_wait_loadcnt 0x0\ns_wait_xcnt 0x0",
        "=v,v", [ptr], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _drain0(x):
    return tl.inline_asm_elementwise(
        "s_wait_storecnt 0x0\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        "=v,v", [x], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def floor_kernel(Plain, Pairs, Cnt, Ep, Big, Stamps, Out, Fail,
                 MODE: tl.constexpr, NSTREAM: tl.constexpr, BIGN: tl.constexpr,
                 ROWS: tl.constexpr, WORDS: tl.constexpr, NCONS: tl.constexpr,
                 DELAY: tl.constexpr):
    pid = tl.program_id(0)
    epoch = tl.load(Ep + pid) + 1
    tl.store(Ep + pid, epoch)
    t0 = _clock()
    PER: tl.constexpr = WORDS // 128
    lane = tl.arange(0, 128)
    if pid < ROWS:
        # producers: start after DELAY sleeps so consumers are already waiting
        for _ in tl.static_range(DELAY):
            tl.inline_asm_elementwise("s_sleep 127\nv_mov_b32 $0, 0", "=v", [], dtype=tl.int32,
                                      is_pure=False, pack=1)
        ts = _clock()
        z = lane * 0
        for k in tl.static_range(PER):
            w = pid * WORDS + k * 128 + lane
            val = epoch * 7 + w
            if MODE == 0:
                z += _st32(Plain + w, val)
            else:
                pair = (epoch.to(tl.int64) << 32) | (val.to(tl.int64) & 0xFFFFFFFF)
                if MODE == 1:
                    z += _st64(Pairs + w, pair)
                else:
                    for rep in tl.static_range(8):
                        z += _st64(Pairs + rep * ROWS * WORDS + w, pair)
        if MODE == 0:
            zz = _drain0(z)
            tl.debug_barrier()
            tl.atomic_add(Cnt + lane * 32 + tl.max(zz, 0), 1, mask=lane < 32, sem="relaxed",
                          scope="gpu")
        tp = _clock()
        tl.store(Stamps + pid * 4 + tl.arange(0, 4),
                 tl.where(tl.arange(0, 4) == 0, ts, tl.where(tl.arange(0, 4) == 1, tp, 0)).to(tl.int64))
    elif pid < ROWS + NCONS:
        c = pid - ROWS
        cacc = lane * 0
        if MODE == 0:
            target = ROWS * epoch
            cnt = _ld32(Cnt + (c % 32) * 32 + lane * 0)
            n = 0
            while (tl.min(cnt, 0) < target) & (n < 20000):
                cnt = _ld32(Cnt + (c % 32) * 32 + lane * 0)
                n += 1
            dep = tl.minimum(tl.min(cnt, 0) - target, 0)
            for r in tl.static_range(ROWS):
                for k in tl.static_range(PER):
                    cacc += tl.load(Plain + dep + r * WORDS + k * 128 + lane)
        else:
            base = Pairs
            if MODE == 2:
                base = Pairs + (c % 8) * ROWS * WORDS
            for r in tl.static_range(ROWS):
                for k in tl.static_range(PER):
                    p = _ld64(base + r * WORDS + k * 128 + lane)
                    n = 0
                    while (tl.min((p >> 32).to(tl.int32), 0) != epoch) & (n < 20000):
                        p = _ld64(base + r * WORDS + k * 128 + lane)
                        n += 1
                    cacc += (p & 0xFFFFFFFF).to(tl.int32)
        td = _clock()
        total = tl.sum(cacc, 0)
        want = 7 * epoch * ROWS * WORDS + (ROWS * WORDS // 2) * (ROWS * WORDS - 1)
        tl.store(Out + c, total)
        tl.store(Fail + c, tl.load(Fail + c) + tl.where(total != want, 1, 0))
        tl.store(Stamps + pid * 4 + tl.arange(0, 4),
                 tl.where(tl.arange(0, 4) == 2, td, tl.where(tl.arange(0, 4) == 3, t0, 0)).to(tl.int64))
    else:
        # streamers: read a big buffer for the whole launch (simulated weights)
        s = pid - ROWS - NCONS
        sacc = tl.zeros([1024], tl.int32)
        i = tl.arange(0, 1024)
        for k in range(0, BIGN // (NSTREAM * 1024)):
            sacc += tl.load(Big + (k * NSTREAM + s) * 1024 + i, cache_modifier=".cg")
        tl.store(Out + 4096 + s, tl.sum(sacc, 0))


def main():
    modes = [int(m) for m in sys.argv[1].split(",")]
    nstream = int(sys.argv[2])
    ncons = 72
    big = torch.ones(256 * 1024 * 1024, device="cuda", dtype=torch.int32)  # 1 GiB
    res = []
    for mode in modes:
        plain = torch.zeros(ROWS * WORDS, device="cuda", dtype=torch.int32)
        pairs = torch.zeros(8 * ROWS * WORDS, device="cuda", dtype=torch.int64)
        cnt = torch.zeros(32 * 32, device="cuda", dtype=torch.int32)
        grid = ROWS + ncons + nstream
        ep = torch.zeros(grid, device="cuda", dtype=torch.int32)
        stamps = torch.zeros(grid * 4, device="cuda", dtype=torch.int64)
        out = torch.zeros(8192, device="cuda", dtype=torch.int32)
        fail = torch.zeros(ncons, device="cuda", dtype=torch.int32)
        lat = []
        for it in range(12):
            floor_kernel[(grid,)](plain, pairs, cnt, ep, big, stamps, out, fail, MODE=mode,
                                  NSTREAM=max(nstream, 1), BIGN=big.numel(), ROWS=ROWS,
                                  WORDS=WORDS, NCONS=ncons, DELAY=4, num_warps=4)
            torch.cuda.synchronize()
            st = stamps.view(grid, 4).double()
            prod_start = st[:ROWS, 0].min()
            prod_pub = st[:ROWS, 1].max()
            cons_done = st[ROWS:ROWS + ncons, 2]
            if it >= 2:
                lat.append(dict(publish=float(prod_pub - prod_start) / 100,
                                med=float(cons_done.median() - prod_start) / 100,
                                last=float(cons_done.max() - prod_start) / 100))
        f = lambda k: round(sorted(x[k] for x in lat)[len(lat) // 2], 2)
        res.append(dict(mode=mode, streamers=nstream, producer_publish_us=f("publish"),
                        consumer_median_done_us=f("med"), consumer_last_done_us=f("last"),
                        fails=int(fail.sum())))
        print(json.dumps(res[-1]), flush=True)


if __name__ == "__main__":
    main()

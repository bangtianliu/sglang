"""MK.7a: 4 real GPUs, one-shot all-reduce through shared pinned host memory.

No GPU peer/P2P/IPC: every rank maps the same /dev/shm pages (hipHostRegister) and
uses SCOPE_SYS stores/loads. 8 CTAs (one per row) x 8192 BF16, as the S3 AR.
Checks exact FP32 sums (integer-valued inputs) on every rank for many epochs, and
reports in-kernel time and graph-replay time.
Launch with torchrun --nproc-per-node=4 and NCCL_P2P_DISABLE=1.
"""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

assert os.environ.get("NCCL_P2P_DISABLE") == "1", "fabric must stay disabled"
RANK = int(os.environ["LOCAL_RANK"])
os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = str(RANK)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hostmem import HostShared  # noqa: E402

WIDTH = int(os.environ.get("AR_WIDTH", "8192"))  # payload per row; 8 rows (MK.49: 4x/8x sizes)
WORDS = WIDTH // 2  # bf16 pairs per row


@triton.jit
def _clock():
    return tl.inline_asm_elementwise(
        "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\ns_wait_kmcnt 0",
        "=s", [], dtype=tl.uint64, is_pure=False, pack=1)


@triton.jit
def _st_sys(ptr, v):
    return tl.inline_asm_elementwise(
        "global_store_b32 $1, $2, off scope:SCOPE_SYS\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        "=&v,v,v", [ptr, v], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ld_sys(ptr):
    return tl.inline_asm_elementwise(
        "global_load_b32 $0, $1, off scope:SCOPE_SYS\ns_wait_loadcnt 0x0\ns_wait_xcnt 0x0",
        "=v,v", [ptr], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _drain0(x):
    return tl.inline_asm_elementwise(
        "s_wait_storecnt 0x0\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        "=v,v", [x], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _st128_sys(ptr, a, b, c, d):
    return tl.inline_asm_elementwise(
        "global_store_b128 $1, v[2:5], off scope:SCOPE_SYS\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        "=&v,v,{v2},{v3},{v4},{v5}", [ptr, a, b, c, d], dtype=tl.int32, is_pure=False, pack=1)


_LD4_CONS = tl.constexpr(",".join(["={v%d}" % r for r in range(8, 24)] + ["v"] * 4))


@triton.jit
def _ld128x4_sys(p0, p1, p2, p3):
    asm: tl.constexpr = (
        "global_load_b128 v[8:11], $16, off scope:SCOPE_SYS\n"
        "global_load_b128 v[12:15], $17, off scope:SCOPE_SYS\n"
        "global_load_b128 v[16:19], $18, off scope:SCOPE_SYS\n"
        "global_load_b128 v[20:23], $19, off scope:SCOPE_SYS\n"
        "s_wait_loadcnt 0x0\ns_wait_xcnt 0x0")
    return tl.inline_asm_elementwise(asm, _LD4_CONS, [p0, p1, p2, p3], dtype=(tl.int32,) * 16,
                                     is_pure=False, pack=1)


@triton.jit
def _bf(w):
    lo = (w & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    hi = ((w >> 16) & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    return lo, hi


@triton.jit
def _ld_sys_m(ptr, mask):
    saved, v = tl.inline_asm_elementwise(
        "v_mov_b32 $1, 0\nv_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\n"
        "global_load_b32 $1, $2, off scope:SCOPE_SYS\ns_wait_loadcnt 0\ns_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        "=&s,=&v,v,v,~{scc}", [ptr, mask.to(tl.int32)], dtype=(tl.int32, tl.int32),
        is_pure=False, pack=1)
    return v


@triton.jit
def _st_sys_m(ptr, v, mask):
    return tl.inline_asm_elementwise(
        "v_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\nglobal_store_b32 $1, $2, off scope:SCOPE_SYS\n"
        "s_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        "=&s,v,v,v,~{scc}", [ptr, v, mask.to(tl.int32)], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def host_ar(Partial, Pay, Flags, Ep, Out, Stamps, Fail,
            RANK: tl.constexpr, WORDS: tl.constexpr, C: tl.constexpr, BOUND: tl.constexpr):
    """grid = 8 rows x S slices, S = WORDS / (512 C); each thread owns C x 4 words."""
    S: tl.constexpr = WORDS // (512 * C)
    pid = tl.program_id(0)
    row = pid // S
    sl = pid % S
    epoch = tl.load(Ep + pid) + 1
    tl.store(Ep + pid, epoch)
    t0 = _clock()
    lane = tl.arange(0, 128)
    parity = epoch % 2
    z = lane * 0
    for c in tl.static_range(C):
        word = sl * (WORDS // S) + c * 512 + lane * 4
        base = Partial + row * WORDS + word
        z += _st128_sys(Pay + ((parity * 4 + RANK) * 8 + row) * WORDS + word, tl.load(base),
                        tl.load(base + 1), tl.load(base + 2), tl.load(base + 3))
    zz = tl.max(_drain0(z), 0)
    tl.debug_barrier()
    _st_sys_m(Flags + ((RANK * 8 + row) * S + sl) * 32 + lane * 0 + zz, epoch + lane * 0, lane == 0)
    t1 = _clock()
    src = lane % 4
    fptr = Flags + ((src * 8 + row) * S + sl) * 32
    seen = _ld_sys_m(fptr, lane < 4)
    n = 0
    while (tl.min(tl.where(lane < 4, seen - epoch, 0), 0) < 0) & (n < BOUND):
        seen = _ld_sys_m(fptr, lane < 4)
        n += 1
    ok = tl.min(tl.where(lane < 4, seen - epoch, 0), 0) >= 0
    t2 = _clock()
    for c in tl.static_range(C):
        word = sl * (WORDS // S) + c * 512 + lane * 4
        pb = Pay + (parity * 4 * 8 + row) * WORDS + word
        r = _ld128x4_sys(pb, pb + 8 * WORDS, pb + 16 * WORDS, pb + 24 * WORDS)
        ob = Out + row * 2 * WORDS + 2 * word
        for k in tl.static_range(4):
            a = tl.zeros([128], tl.float32)
            b = tl.zeros([128], tl.float32)
            for s in tl.static_range(4):
                lo, hi = _bf(r[s * 4 + k])
                a += lo
                b += hi
            tl.store(ob + 2 * k, a)
            tl.store(ob + 2 * k + 1, b)
    t3 = _clock()
    tl.store(Fail + pid, tl.load(Fail + pid) + tl.where(ok, 0, 1))
    k = tl.arange(0, 4)
    tl.store(Stamps + pid * 4 + k,
             tl.where(k == 0, t1 - t0, tl.where(k == 1, t2 - t1, tl.where(k == 2, t3 - t2, n))).to(tl.int64))


def main():
    out_dir = Path(sys.argv[1])
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))

    def barrier():
        dist.monitored_barrier(timeout=timedelta(seconds=20))

    pay_words = 2 * 4 * 8 * WORDS
    shared = HostShared(f"muse_mk7a_{os.getppid()}", (pay_words + 4 * 8 * 64 * 32) * 4, RANK, barrier)
    if RANK == 0:
        shared.zero()
    barrier()
    pay = shared.tensor(0, pay_words)
    flags = shared.tensor(pay_words * 4, 4 * 8 * 64 * 32)
    gen = torch.Generator().manual_seed(100 + RANK)
    parts = [(torch.randint(-64, 64, (8, WIDTH), generator=gen).float()).bfloat16().cuda()
             for _ in range(2)]
    C = int(os.environ.get("CHUNKS", "8"))
    S = WORDS // (512 * C)
    ep = torch.zeros(8 * S, device="cuda", dtype=torch.int32)
    out = torch.zeros(8, WIDTH, device="cuda", dtype=torch.float32)
    stamps = torch.zeros(8 * S * 4, device="cuda", dtype=torch.int64)
    fail = torch.zeros(8 * S, device="cuda", dtype=torch.int32)

    def expect(k):
        tot = torch.zeros(8, WIDTH)
        for r in range(4):
            g = torch.Generator().manual_seed(100 + r)
            ps = [torch.randint(-64, 64, (8, WIDTH), generator=g).float() for _ in range(2)]
            tot += ps[k]
        return tot

    want = [expect(0).cuda(), expect(1).cuda()]

    def run(k):
        host_ar[(8 * S,)](parts[k].view(torch.int32), pay, flags, ep, out, stamps, fail,
                          RANK=RANK, WORDS=WORDS, C=C, BOUND=20000, num_warps=4)

    run(0)  # compile on every rank before any timed or checked epoch
    torch.cuda.synchronize()
    barrier()
    run(1)
    torch.cuda.synchronize()
    fail.zero_()

    # eager epochs alternate inputs; every one must be exact on every rank
    bad = 0
    for it in range(40):
        barrier()
        run(it % 2)
        torch.cuda.synchronize()
        bad += int((out != want[it % 2]).sum())
    st = stamps.view(8 * S, 4).double()
    print(json.dumps(dict(rank=RANK, eager_mismatch=bad, fails=int(fail.sum()),
                          publish_us=float(st[:, 0].max()) / 100,
                          wait_us=float(st[:, 1].max()) / 100,
                          read_us=float(st[:, 2].max()) / 100,
                          polls=int(st[:, 3].max()))), flush=True)
    # graph timing
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(0)
    ts = []
    for it in range(30):
        barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
        bad += int((out != want[0]).sum())
    ts.sort()
    st = stamps.view(8 * S, 4).double()
    rec = dict(rank=RANK, slices=S, graph_us_median=round(ts[len(ts) // 2], 2), graph_us_min=round(ts[0], 2),
               mismatch_total=bad, fails=int(fail.sum()),
               last_publish_us=float(st[:, 0].max()) / 100, last_wait_us=float(st[:, 1].max()) / 100,
               last_read_us=float(st[:, 2].max()) / 100)
    print(json.dumps(rec), flush=True)
    # steady state: 20 back-to-back ARs in one graph (ranks lock-step after the first)
    g20 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g20):
        for _ in range(20):
            run(0)
    t20 = []
    for it in range(10):
        barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        g20.replay()
        e.record()
        torch.cuda.synchronize()
        t20.append(s.elapsed_time(e) * 1000)
        bad += int((out != want[0]).sum())
    t20.sort()
    st = stamps.view(8 * S, 4).double()
    rec.update(per_ar_us_20chain=round(t20[len(t20) // 2] / 20, 2), mismatch_total=bad,
               fails=int(fail.sum()), chain_publish_us=float(st[:, 0].max()) / 100,
               chain_wait_us=float(st[:, 1].max()) / 100, chain_read_us=float(st[:, 2].max()) / 100)
    print(json.dumps(rec), flush=True)
    (out_dir / f"rank-{RANK}.json").write_text(json.dumps(rec) + "\n")
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""GPU0: visibility latency of a tagged pair (gfx950 mailbox style, no drain, no flag).
CTA i stores pair (tag<<32 | i) to slot i at realtime t0; CTA (i + S) % G polls slot i with
one lane (bounded) and records when it sees the tag. Realtime clock is device-wide."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from handoff_floor import _clock, _ld64, _st64  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402


@triton.jit
def _ld64_sys(ptr):
    return tl.inline_asm_elementwise(
        "global_load_b64 $0, $1, off scope:SCOPE_SYS\ns_wait_loadcnt 0x0\ns_wait_xcnt 0x0",
        "=v,v", [ptr], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _st64_se(ptr, v):
    return tl.inline_asm_elementwise(
        "global_store_b64 $1, $2, off scope:SCOPE_SE\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        "=&v,v,v", [ptr, v], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _st64_one(ptr, v, mask):
    """SCOPE_DEV 64-bit store from the lanes where mask != 0 only."""
    return tl.inline_asm_elementwise(
        "v_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\n"
        "global_store_b64 $1, $2, off scope:SCOPE_DEV\ns_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        "=&s,v,v,v,~{scc}", [ptr, v, mask], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ld64_one(ptr, mask):
    """SCOPE_DEV 64-bit load on the lanes where mask != 0 only (others return 0)."""
    saved, value = tl.inline_asm_elementwise(
        "v_mov_b64 $1, 0\nv_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\n"
        "global_load_b64 $1, $2, off scope:SCOPE_DEV\ns_wait_loadcnt 0x0\n"
        "s_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        "=&s,=&v,v,v,~{scc}", [ptr, mask], dtype=(tl.int32, tl.int64), is_pure=False, pack=1)
    return value


@triton.jit
def pair_kernel(Pairs, Ep, Stamps, G: tl.constexpr, S: tl.constexpr, ST: tl.constexpr,
                LD: tl.constexpr, LOAD: tl.constexpr):
    pid = tl.program_id(0)
    t = tl.arange(0, 32)
    epoch = tl.load(Ep + pid) + 1
    tl.store(Ep + pid, epoch)
    src = (pid - S + G) % G
    # consumers start polling first; producers wait ~10 us (s_sleep) then store
    for _ in tl.static_range(8):
        tl.inline_asm_elementwise("s_sleep 127\nv_mov_b32 $0, 0", "=v", [], dtype=tl.int32,
                                  is_pure=False, pack=1)
    pair = (epoch.to(tl.int64) << 32) | pid.to(tl.int64)
    t0 = _clock()
    if ST == 3:
        _st64_one(Pairs + pid + t * 0, pair + t * 0, (t == 0).to(tl.int32))  # ONE lane
    elif ST == 0:
        _st64(Pairs + pid + t * 0, pair + t * 0)          # SCOPE_DEV store, no drain
    elif ST == 1:
        tl.atomic_xchg(Pairs + pid + t * 0, pair + t * 0, mask=t == 0, sem="relaxed", scope="gpu")
    else:
        _st64_se(Pairs + pid + t * 0, pair + t * 0)
    # poll the source's pair (one lane matters; all 32 lanes load the same word)
    one = t == 0
    if LD == 3:
        p = tl.max(_ld64_one(Pairs + src + t * 0, one.to(tl.int32)), 0)
    elif LD == 0:
        p = tl.max(_ld64(Pairs + src + t * 0), 0)
    elif LD == 1:
        p = tl.max(_ld64_sys(Pairs + src + t * 0), 0)
    else:
        p = tl.max(tl.where(one, tl.atomic_add(Pairs + src + t * 0, 0, mask=one, sem="relaxed",
                                               scope="gpu"), 0), 0)
    n = 0
    while ((p >> 32).to(tl.int32) != epoch) & (n < 200000):
        if LD == 3:
            p = tl.max(_ld64_one(Pairs + src + t * 0, one.to(tl.int32)), 0)
        elif LD == 0:
            p = tl.max(_ld64(Pairs + src + t * 0), 0)
        elif LD == 1:
            p = tl.max(_ld64_sys(Pairs + src + t * 0), 0)
        else:
            p = tl.max(tl.where(one, tl.atomic_add(Pairs + src + t * 0, 0, mask=one, sem="relaxed",
                                                   scope="gpu"), 0), 0)
        n += 1
    t1 = _clock()
    # optional payload: read LOAD words the source published (none here)
    q = tl.arange(0, 4)
    tl.store(Stamps + pid * 4 + q, tl.where(q == 0, t0, tl.where(q == 1, t1, tl.where(q == 2, n, p & 0xFFFFFFFF))).to(tl.int64))


def main():
    require_gpu0()
    for G in (192,):
        for ST, LD in ((0, 0), (3, 3), (3, 2), (1, 3), (1, 2)):
            for S in (0, 1, 8):
                pairs = torch.zeros(G, device="cuda", dtype=torch.int64)
                ep = torch.zeros(G, device="cuda", dtype=torch.int32)
                st = torch.zeros(G * 4, device="cuda", dtype=torch.int64)
                lat, timeouts = [], 0
                for _ in range(5):
                    pair_kernel[(G,)](pairs, ep, st, G, S, ST, LD, 0, num_warps=4)
                    torch.cuda.synchronize()
                    v = st.view(G, 4).cpu()
                    t0 = v[:, 0].double() * 0.01
                    t1 = v[:, 1].double() * 0.01
                    src = (torch.arange(G) - S) % G
                    lat.append(t1 - t0[src])
                    timeouts += int((v[:, 2] >= 200000).sum())
                lat = torch.cat(lat)
                print(json.dumps(dict(grid=G, store=["dev_32_lanes_same_addr", "atomic_xchg_1_lane", "se_store", "dev_1_lane"][ST],
                                      load=["dev_32_lanes", "sys", "atomic_add0_1_lane", "dev_1_lane"][LD], shift=S,
                                      visible_med_us=round(float(lat.median()), 2),
                                      visible_p90_us=round(float(lat.quantile(0.9)), 2),
                                      polls_med=int(v[:, 2].median()),
                                      timeouts=timeouts)), flush=True)


if __name__ == "__main__":
    main()

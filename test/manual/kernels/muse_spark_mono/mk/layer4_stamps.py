"""MK.43 probe: layer4_ab with a 1-CTA realtime stamp kernel after every Triton kernel launch.
The 10-layer graph is the last capture, so its slots hold the last replay: per-kernel time =
difference of consecutive stamps (includes the stamp launch, the same for every variant).
Run with one VARIANTS entry and one CTX."""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer4_ab  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402


@triton.jit
def stamp(Buf, slot):
    t = tl.inline_asm_elementwise(
        "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\ns_wait_kmcnt 0", "=s", [],
        dtype=tl.uint64, is_pure=False, pack=1)
    tl.store(Buf + slot, t.to(tl.int64))


@triton.jit
def idle(Buf, N: tl.constexpr):
    """One CTA that sleeps about N x 0.65 us (s_sleep 127 = 127 x 64 clocks) and does nothing."""
    acc = 0
    for _ in tl.static_range(N):
        acc += tl.inline_asm_elementwise("s_sleep 127\nv_mov_b32 $0, 0", "=v", [],
                                         dtype=tl.int32, is_pure=False, pack=1)
    tl.store(Buf, tl.load(Buf) + acc * 0)


# Probes (env): STAMP2=1 -> two stamps after every kernel: "body" = kernel + first stamp,
# "gap" = the second (trivial) stamp alone at that position in the graph.
# GAP_BEFORE / GAP_AFTER = comma list of kernel names: an idle kernel of GAP_N sleeps there.
STAMP2 = os.environ.get("STAMP2") == "1"
GAP_BEFORE = set(filter(None, os.environ.get("GAP_BEFORE", "").split(",")))
GAP_AFTER = set(filter(None, os.environ.get("GAP_AFTER", "").split(",")))
GAP_N = int(os.environ.get("GAP_N", "32"))
idle_buf = torch.zeros(64, device="cuda", dtype=torch.int64)
SLOTS = 1 << 16
buf = torch.zeros(SLOTS, device="cuda", dtype=torch.int64)
seq = []  # (slot, kernel name) in launch order
orig = JITFunction.run


def run(self, *args, grid, warmup, **kwargs):
    probe = not warmup and self is not stamp and self is not idle
    name = self.fn.__name__
    if probe and name in GAP_BEFORE:
        orig(idle, idle_buf, GAP_N, grid=(1,), warmup=False, num_warps=1)
        seq.append("idle_before_" + name)
        orig(stamp, buf, (len(seq) - 1) % SLOTS, grid=(1,), warmup=False, num_warps=1)
    r = orig(self, *args, grid=grid, warmup=warmup, **kwargs)
    if probe:
        seq.append(name)
        orig(stamp, buf, (len(seq) - 1) % SLOTS, grid=(1,), warmup=False, num_warps=1)
        if STAMP2:
            seq.append("gap_after_" + name)
            orig(stamp, buf, (len(seq) - 1) % SLOTS, grid=(1,), warmup=False, num_warps=1)
        if name in GAP_AFTER:
            orig(idle, idle_buf, GAP_N, grid=(1,), warmup=False, num_warps=1)
            seq.append("idle_after_" + name)
            orig(stamp, buf, (len(seq) - 1) % SLOTS, grid=(1,), warmup=False, num_warps=1)
    return r


JITFunction.run = run
layer4_ab.main()
torch.cuda.synchronize()
JITFunction.run = orig
names = seq
v = buf.cpu().double()
med = lambda x: round(sorted(x)[len(x) // 2], 2)  # noqa: E731
variants = os.environ["VARIANTS"].split(",")
# tail of the launch sequence: per variant [single layer][10 layers], in variant order
end = len(names)
out = {}
for var in reversed(variants):
    head = names[end - 1]  # last kernel of a layer
    idx = [i for i in range(end) if names[i] == head]
    per = idx[-1] - idx[-2]
    start = end - 10 * per
    dur = defaultdict(list)
    for layer in range(1, 10):
        for k in range(per):
            i = start + layer * per + k
            dur[(k, names[i])].append(float(v[i % SLOTS] - v[(i - 1) % SLOTS]) / 100)
    rows = [(k, n, med(d)) for (k, n), d in sorted(dur.items())]
    out[var] = dict(per_layer=per, layer_us=round(sum(r[2] for r in rows), 2),
                    kernels=[[n, t] for _, n, t in rows])
    if os.environ.get("STAMP_LAYERS") == "1":
        # every layer of the last 10-layer replay, layer 0 included (its first kernel is
        # measured from the previous stamp, i.e. across the graph boundary: skip k == 0 there)
        out[var]["layers"] = [[names[start + layer * per + k],
                               [None if layer == 0 and k == 0 else
                                round(float(v[(start + layer * per + k) % SLOTS]
                                            - v[(start + layer * per + k - 1) % SLOTS]) / 100, 1)
                                for layer in range(10)]] for k in range(per)]
    end = start - per  # skip this variant's single-layer capture
print(json.dumps(dict(rank=int(os.environ["LOCAL_RANK"]), stamps=out)), flush=True)

"""GPU0 tagged-pair atomicity, unsigned tags, stale and missing producers."""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx
from layer_fixture import require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_tagged as mailbox,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
)


@triton.jit
def probe(
    Pairs,
    Out,
    Epoch,
    Failure,
    Record,
    Counts,
    MISSING: tl.constexpr = False,
    DELAY: tl.constexpr = False,
):
    pid = tl.program_id(0)
    i = tlx.require_layout(tl.arange(0, 512), ROW1)
    tag = tl.load(Epoch).to(tl.uint32)
    if pid < 2:
        if not (MISSING and pid == 0):
            if DELAY:
                for _ in range(256):
                    tl.inline_asm_elementwise(
                        "s_sleep 1\nv_mov_b32 $0,0", "=v", [], tl.int32, False, 1
                    )
            index = pid * 512 + i
            word = (index.to(tl.uint32) * 1664525) ^ tag ^ 0xDEADBEEF
            mailbox.put(Pairs + index, word, tag, i < 512)
    else:
        index = (pid - 2) * 512 + i
        rec = Record + (pid * 4 + (i // 32) % 4) * 8
        value, retries = mailbox.poll(
            Pairs + index, tag, i < 512, Failure, rec, index, LIMIT=512
        )
        if tl.load(Failure, cache_modifier=".cg", volatile=True) == 0:
            tl.store(Out + index, value)
        tl.store(Counts + (pid - 2) * 512 + i, retries)


def main():
    require_gpu0()
    p = torch.zeros(1024, dtype=torch.int64, device="cuda")
    out = torch.zeros(1024, dtype=torch.int32, device="cuda")
    epoch = torch.ones((), dtype=torch.int32, device="cuda")
    failure = torch.zeros(2, dtype=torch.int32, device="cuda")
    rec = torch.zeros((4, 4, 8), dtype=torch.int32, device="cuda")
    counts = torch.zeros(1024, dtype=torch.int32, device="cuda")
    logs = []
    # Full current epoch must match while previous epoch payload remains present.
    for tag in (1, 2, 0x80000001, 0xFFFFFFFF, 3):
        epoch.fill_(tag if tag < 2**31 else tag - 2**32)
        k = probe[(4,)](p, out, epoch, failure, rec, counts, DELAY=True, num_warps=4)
        torch.cuda.synchronize()
        assert int(failure[0]) == 0, rec.cpu().tolist()
        index = torch.arange(1024, dtype=torch.int64, device="cuda")
        expected = ((index * 1664525) ^ tag ^ 0xDEADBEEF) & 0xFFFFFFFF
        torch.testing.assert_close(
            out.to(torch.int64) & 0xFFFFFFFF, expected, rtol=0, atol=0
        )
        logs.append(
            dict(
                tag=tag,
                counts=torch.unique(counts, return_counts=True)[0].cpu().tolist(),
                max_retry=int(counts.max()),
            )
        )
        asm = k.asm["amdgcn"]
        assert "global_wb" not in asm and "global_inv" not in asm
        assert (
            "global_store_b64" in asm
            and "global_load_b64" in asm
            and "s_wait_xcnt" in asm
        )
    epoch.fill_(4)
    out.fill_(-1)
    probe[(4,)](p, out, epoch, failure, rec, counts, MISSING=True, num_warps=4)
    torch.cuda.synchronize()
    assert int(failure[0]) == 5, (failure.cpu().tolist(), rec.cpu().tolist())
    records = rec.cpu().reshape(-1, 8)
    late = records[records[:, 0] == 4]
    assert (
        len(late) > 0
        and bool((late[:, 1] == 3).all())
        and bool((late[:, 3] == 512).all())
    ), late
    assert bool((out[:512] == -1).all())
    result = dict(
        normal=logs, missing_fail_stop=True, stale_records=late.tolist(), trap=False
    )
    Path("/home/bangtliu/coord/logs/phaseT13-mailbox-proof.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for ext in ("ttgir", "llir", "amdgcn"):
        Path("/home/bangtliu/coord/logs/phaseT13-mailbox-proof." + ext).write_text(
            k.asm[ext]
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

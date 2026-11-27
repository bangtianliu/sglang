"""CPU-only old-vs-round-trip-rule S3 boundary compilation; no store/index edits."""

import hashlib
import json
import struct
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.collective_fast import (
    oneshot_plan,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_attn_tail as s,
)

root = Path("/home/bangtliu/coord/logs/phaseT-ir/T4-S3-compile")
root.mkdir(exist_ok=True)
p = oneshot_plan(8192, 8, 4)
payload, flags = [
    (r.base, r.rank_stride, r.parity_stride)
    for r in [p.region("payload"), p.region("flags")]
]
sig = dict(
    Partial="*bf16",
    Stream="*fp32",
    Pre="*bf16",
    Alpha="*fp32",
    Beta="*fp32",
    Out="*bf16",
    Updated="*fp32",
    Status="*i32",
    Epochs="*i32",
    Scratch="*i32",
    Peers="*i64",
    Reduced="*fp32",
    Failure="*i32",
)
hashes = []
for rule in (False, True):
    const = dict(
        GRID=8,
        RANK=0,
        PAYLOAD=payload,
        FLAGS=flags,
        Seed=None,
        PIN_LAYOUT=True,
        CARRIER_RULE=rule,
    )
    src = ASTSource(
        s.s3_boundary,
        signature=sig,
        constexprs=const,
        attrs={
            (i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
            for i in range(len(sig))
        },
    )
    k = triton.compile(
        src,
        target=GPUTarget("hip", "gfx1250", 32),
        options=dict(num_warps=4, enable_fp_fusion=False),
    )
    folder = root / ("rule" if rule else "baseline")
    folder.mkdir(exist_ok=True)
    for ext in ("ttir", "ttgir", "llir", "amdgcn"):
        (folder / f"tlx.{ext}").write_text(k.asm[ext])
    blob = k.asm["hsaco"]
    (folder / "tlx.hsaco").write_bytes(blob)
    shoff = struct.unpack_from("<Q", blob, 40)[0]
    entsize, number, names_idx = struct.unpack_from("<HHH", blob, 58)
    off, size = struct.unpack_from("<QQ", blob, shoff + names_idx * entsize + 24)
    names = blob[off : off + size]
    for index in range(number):
        start = shoff + index * entsize
        name_at = struct.unpack_from("<I", blob, start)[0]
        name = names[name_at:].split(b"\0", 1)[0]
        if name == b".text":
            off, size = struct.unpack_from("<QQ", blob, start + 24)
            hashes.append(hashlib.sha256(blob[off : off + size]).hexdigest())
    asm = k.asm["amdgcn"]
    print(
        json.dumps(
            dict(
                rule=rule,
                lds=k.metadata.shared,
                ds_load_b128=asm.count("ds_load_b128"),
                ds_store_b128=asm.count("ds_store_b128"),
                barriers=asm.count("s_barrier_signal"),
                resources=[
                    l.strip()
                    for l in asm.splitlines()
                    if any(
                        x in l
                        for x in [
                            ".vgpr_count:",
                            ".vgpr_spill_count:",
                            ".sgpr_spill_count:",
                            ".private_segment_fixed_size:",
                        ]
                    )
                ],
            )
        ),
        flush=True,
    )

assert len(hashes) == 2 and hashes[0] == hashes[1], hashes
print(json.dumps(dict(machine_text_identical=True, sha256=hashes[0])))

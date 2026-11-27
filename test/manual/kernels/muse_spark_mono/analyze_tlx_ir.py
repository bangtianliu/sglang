"""Static IR/asm inventory and literal diffs; counts are static, not profiler counters."""

import collections
import difflib
import json
import re
from pathlib import Path

ROOT = Path("/home/bangtliu/coord/logs/phaseT-ir")


def analyze(stage, side):
    path = ROOT / stage
    texts = {
        ext: (path / f"{side}.{ext}").read_text() for ext in ("ttgir", "llir", "amdgcn")
    }
    asm = texts["amdgcn"]
    ops = collections.Counter()
    for line in asm.splitlines():
        m = re.match(r"^\s*([a-z][a-z0-9_]+)(?:\s|$)", line)
        if m:
            ops[m[1]] += 1
    resource = json.loads((path / f"{side}.json").read_text())
    for field, pattern in [
        ("sgprs", r"\.amdhsa_next_free_sgpr\s+(\d+)"),
        ("vgprs_directive", r"\.amdhsa_next_free_vgpr\s+(\d+)"),
    ]:
        m = re.search(pattern, asm)
        resource[field] = int(m[1]) if m else None
    blocks = re.findall(r"#ASMSTART(.*?)#ASMEND", asm, re.S)
    scoped_blocks = [
        b
        for b in blocks
        if "scope:SCOPE_DEV" in b and ("global_load_" in b or "global_store_" in b)
    ]
    if stage in ("handoff", "xcnt"):
        assert scoped_blocks, (stage, side, "no scoped asm blocks parsed")
    assert all("s_wait_xcnt" in b for b in scoped_blocks), (
        stage,
        side,
        "missing XCNT in scoped VMEM asm",
    )
    ir = texts["ttgir"]
    return dict(
        resources=resource,
        opcodes=dict(sorted(ops.items())),
        memory_ops={
            k: v
            for k, v in ops.items()
            if k.startswith(("global_", "buffer_", "tensor_", "ds_"))
        },
        sync_ops={
            k: v for k, v in ops.items() if k.startswith(("s_wait_", "s_barrier"))
        },
        matrix_ops={k: v for k, v in ops.items() if k.startswith("v_wmma_")},
        convert_ops={k: v for k, v in ops.items() if k.startswith("v_cvt_")},
        policies=dict(
            scope_DEV=asm.count("scope:SCOPE_DEV"),
            TH_fields=collections.Counter(re.findall(r"th:([A-Z_]+)", asm)),
        ),
        scoped_asm_blocks=len(scoped_blocks),
        scoped_blocks_xcnt=True,
        layouts=[
            l
            for l in ir.splitlines()
            if l.startswith("#")
            and any(
                x in l
                for x in (
                    "#ttg.blocked",
                    "#ttg.amd_wmma",
                    "#ttg.padded",
                    "#ttg.swizzled",
                )
            )
        ],
        tdm_waits=re.findall(r"amdg.async_tdm_wait\s+\{num = (\d+)", ir),
        dot_count=len(re.findall(r"= tt.dot ", ir)),
        split_tree_lines=[l.strip() for l in ir.splitlines() if "arith.addf" in l],
        llvm_symbol_references=collections.Counter(
            re.findall(r"@([a-zA-Z0-9_.]+)", texts["llir"])
        ),
    )


def main():
    result = {}
    for stage in ("handoff", "xcnt", "projection", "expert-dot"):
        result[stage] = {side: analyze(stage, side) for side in ("gluon", "tlx")}
        for ext in ("ttgir", "llir", "amdgcn"):
            a = (ROOT / stage / f"gluon.{ext}").read_text().splitlines(keepends=True)
            b = (ROOT / stage / f"tlx.{ext}").read_text().splitlines(keepends=True)
            (ROOT / stage / f"gluon-vs-tlx.{ext}.diff").write_text(
                "".join(
                    difflib.unified_diff(
                        a, b, fromfile=f"gluon.{ext}", tofile=f"tlx.{ext}"
                    )
                )
            )
    for side in ("gluon", "tlx"):
        p = result["projection"][side]
        assert p["matrix_ops"] == {"v_wmma_f32_16x16x32_bf16": 128}
        assert p["memory_ops"]["tensor_load_to_lds"] == 16
        assert p["dot_count"] == 8 and len(p["split_tree_lines"]) == 3
        ir = (ROOT / "projection" / f"{side}.ttgir").read_text()
        dots = re.findall(r"^\s*(%\w+) = tt\.dot [^\n]*?, (%\w+) :", ir, re.M)
        assert len(dots) == 8
        assert all(dots[i][1] == dots[i - 1][0] for i in (1, 3, 5, 7))
        assert len({dots[i][1] for i in (0, 2, 4, 6)}) == 1
        tree = {dots[2 * i + 1][0]: f"p{i}" for i in range(4)}
        for line in p["split_tree_lines"]:
            out, a, b = re.search(r"(%\w+) = arith.addf (%\w+), (%\w+)", line).groups()
            tree[out] = "(" + tree[a] + "+" + tree[b] + ")"
        p["fixed_tree"] = tree[out]
        assert p["fixed_tree"] == "((p0+p2)+(p1+p3))"
        assert result["expert-dot"][side]["matrix_ops"] == {
            "v_wmma_f32_16x16x32_bf16": 4
        }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# T0 static compiler comparison",
        "",
        "All counts are static instruction sites, including column-zero inline asm. They are not dynamic counters. Projection is fully unrolled: divide its matrix/LDS counts by8 for a K512 step. XCNT has two runtime16-round loops; its64 scoped stores/64 scoped loads are sites per corresponding loop body. Handoff counts include both branches and the bounded poll-loop body.",
        "",
        "|Proof|Side|VGPR|SGPR|Spills|ScratchB|LDS B|CTA/CU|WMMA|TDM|LDS loads/stores|Barrier signals|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, sides in result.items():
        for side, d in sides.items():
            r = d["resources"]
            op = d["opcodes"]
            lds = sum(v for k, v in op.items() if k.startswith(("ds_load", "ds_store")))
            lines.append(
                f"|{stage}|{side}|{r['vgprs']}|{r['sgprs']}|{r['spills']}|{r['metadata'].get('global_scratch_size')}|{r['lds']}|{r['occupancy']}|{sum(d['matrix_ops'].values())}|{op.get('tensor_load_to_lds', 0)}|{lds}|{op.get('s_barrier_signal', 0)}|"
            )
    lines += [
        "",
        "## Dataflow and explanations",
        "",
        "- Handoff: payload1024int32, one128B private consumer line, epoch and result/poll records. Both sides issue the same DEV b32 payload/flag VMEM sites and XCNT guards. Poll clock is shader cycles, clock cadence64, max4096polls. Gluon publish drains stores then gl.barrier, whose conservative lowering adds another store wait; TLX explicitly drains then uses LDS-only workgroup_barrier (2 vs1 static store-wait sites). CTA reductions use16B LDS on both sides.",
        "- XCNT: input/output b128 normal transfers; masked b32 DEV Scratch stores followed by XCNT at every asm boundary, and b32 DEV reads with LOADcnt+XCNT. TLX selected blocked sizePerThread[1,1] for inline-load results, then converts to[1,4] for vector stores; TTGIR convert_layout explains32KiB LDS staging and extra barriers/DS operations. Gluon pins[1,4] throughout, requiring no LDS. Normal TLX I/O lowers to buffer ops; Gluon to global ops. Neither normal I/O is a polled peer region. No cache policy change is claimed from opcode spelling alone.",
        "- Projection: X and W use16TDM-to-LDS sites total (8K512steps x2tensors), each16x512BF16 tile. Same padded[512:+8] ring and133088B LDS after the view-transpose correction. TDM scope/cache policy arguments use defaults; no explicit TH_NT overrides. Per step:64ds_load_b128 sites (32A+32B),16WMMA sites; output has2b128 stores plus scalar status. BF16 conversion has4v_cvt_pk_bf16_f32 sites; no FP4/FP8 conversion in this proof.",
        "- The initial register tl.trans(B) introduced per-step local_alloc/local_load conversion, extra16KiB scratch LDS and b16 stores. Using local_trans on the LDS view removes it; preserved initial IR is under logs/phaseT0-projection-r1-*. Final TLX uses auto WMMA warp basis[[1,0]], Gluon explicit[[0,1]]. Both output tiles are16x16; redundant-warp ownership differs without changing arithmetic.",
        "- Both projection TTGIRs have8dot operations, paired into4independent K1024accumulators. Final3addf lines show(p0+p2)+(p1+p3), then BF16 rounding. No reassociation/fusion enabled. Same128WMMA sites in LLVM/asm. Full output bits and changing-input replays match.",
        "- Final projection tensor-wait site counts match20; requested IR waits and compiler-inserted waits are separately visible. TLX source includes an LDS-only workgroup barrier after each tile load to protect bank reuse; conservative local_load/standard-dot scheduling differs from Gluon relaxed operand loads. Final barrier signals17vs13; dscnt sites8vs105 and ALU-wait sites116vs69 reflect different scheduling/scoreboard grouping, not fewer LDS reads. Do not infer cycles from wait counts.",
        "- Projection register allocation is295vs262VGPR, with unchanged zero scratch/spills and occupancy2. Exact attribution of the33register difference or sub-microsecond latency requires further scheduling/liveness analysis; it is not presented as a proven causal breakdown. No stage adoption is proposed.",
        "- Expert-dot: both emit4WMMA16x16x32BF16 sites for the M16N64K128 tile. TLX auto layouts coalesce B as b128 buffer reads rather than the deliberately operand-layout Gluon reference's64u16 sites; TLX adds transposed LDS reads. LDS16384vs8192 and automatic wave/register mapping differ; this probe demonstrates instruction availability and exact small-integer math, not an expert-core speedup.",
        "",
        "Inline asm, flag scope, wait positions, padding, layout ownership, scalar address setup, SGPR/VGPR allocation, source locations/SSA renaming and metadata are retained in literal three-level diffs. Remaining register/scheduling questions block adoption, not the feasibility report. No runtime hardware bank-conflict counter was collected; padding and DS instructions are compiler evidence only.",
    ]
    (ROOT / "comparison.md").write_text("\n".join(lines) + "\n")
    print("Saved static inventory, dataflow explanations and all12 IR diffs.")


if __name__ == "__main__":
    main()

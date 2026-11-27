# Muse Spark gfx1250 manual experiments

Phase 3 stage tests use one gfx1250 GPU with both HIP_VISIBLE_DEVICES=0 and
CUDA_VISIBLE_DEVICES=0, under timeout -s KILL 300. No model construction or
process-group initialization is needed. Multi-GPU experiments remain halted.
The historical P.1 script refuses both entry points unless Claude has written
coord/approvals/<SHA256-of-script>.ok for its exact contents.

`layer_fixture.build_layer` constructs seeded native TP4/EP4 tensors. Select
components from qkvg, o_proj, router, pooler, unpooler, experts and norms; selecting
all materializes one rank's complete layer. Omitted components avoid allocating
irrelevant weights during individual stage tests. Router is input-major FP32;
other dense weights are output-major BF16. Experts are contiguous unshuffled
MXFP4 bytes and E8M0 scales, not AITER shuffled weights. Stored norm gamma is
zero-centered: add the appropriate gain before use. Post-attention norm is
weightless. Post-expert and post-FFN norms apply gain to the input before RMS.
Residual alpha/beta are precomputed FP32 coefficients. Effective gamma+1
weights are available alongside the stored gamma. Selecting both router and
pooler also constructs native joint packed rows (pooler, router high, router
low), with pooler aliasing the packed prefix. Fixtures represent post-load
weights; the checkpoint Q/K row rearrangement has already been accounted for
by generating directly in the native coordinate system.

Cache is NHD FP8 E4M3FN, page16, four KV heads and D64. Positions are logical;
locations are physical slots in a permuted page table. The full-to-SWA map maps
the recent 2048 physical slots bijectively to compact slots and marks old slots
-1. Sliding tests use the first 128 physical cache pages; the larger allocation
lets full and sliding tests share shapes while preserving untouched sentinels.
K/V descales are deliberately different (0.125 and0.25).

`check_layer_fixture.py` checks seed reproducibility, native packed expert
shapes, joint router/pooler packing and nonidentity page/SWA mapping at
8k/32k/128k. `bench_hbm.py` measures Gluon streaming reads and Torch copies at
16/64/256/1024 MiB with random input. This read kernel is a bandwidth measurement
tool, independent of the eventual stage-language decision. Full read reductions
and copy outputs are validated. Working sets exceed the reported4 MiB L2;
100 GPU-event samples surround individual CUDA-graph replays. The1 GiB read
bandwidth supplies the stage-bound denominator. Copy read-equivalent and total
read+write bandwidth are comparison columns, not read ceilings. `layer_traffic.py`
computes compulsory read bytes and all three time estimates in pure host Python.
Recon section j's latest update supersedes its earlier copy-only denominator.

## S1 validation

`repro_dense_partial.py` runs the stock split-K GEMM as the first GPU compute
kernel of a fresh process, one TP4 or TP8 case. Both are incorrect on the
current Triton3.8 image. Investigation findings are in coord/s1-dense-partial.md;
S1 never calls that reference. No production fix or toolchain upgrade.

`check_s1.py` compares the Gluon stage to an independent FP64 oracle, with and
without modeled BF16 rounding, and the real SGLang UnquantizedLinearMethod.apply
projection followed by qk_norm_rope_cache. The local linear owns native weights
and does not initialize distributed state. AITER chooses its default Triton
solution for this untuned projection shape, as recorded in logs.

The stage uses FP32 accumulation, BF16 projection/norm/query scaling/RoPE
rounding, and FP8 cache conversion. It is numerically compatible rather than
bitwise equivalent across GEMM accumulation orders. Test bounds: normalized
L-infinity error below0.005 vs rounded FP64 and0.02 vs ideal FP64; outputs also
pass elementwise BF16 tolerances vs SGLang (rtol0.02,atol0.125 for scaled Q,
0.02 otherwise). Cache must equal rounded FP64 byte for byte on the fixtures;
vs SGLang it must be within one FP8 code, with every untouched byte identical.
Signed zero shares a code-distance origin. Tests record all byte mismatches.
With identical projected inputs, all Q/K/V/G and complete caches must match
the SGLang epilogue bit for bit for all four RoPE/mapping combinations.

Default test coverage: two seeds x three contexts x sliding/full, plus1/37/191
CTA task-stride and negative-location cases. Standalone timing uses192 CTAs,
one fused launch with reused workspace,100 CUDA-graph replay samples and a72MiB weight (larger than4MiB L2).
Performance and errors are recorded beside the streaming-read compulsory-byte
bound and copy estimates. The empirical bandwidth excludes other stage costs.
Assembly and separate projection/epilogue timings are retained in coord/logs.
The stage remains unconnected to production runtime dispatch. The user's latest
instruction permits this Gluon validation while the final language choice is
pending; no uTLX implementation is introduced here.


## Phase3b shared core and one-launch S1

`stages/skinny_gemm.py` is the shared M8 BF16 device core for QKVG/joint
router-pooler(4608,8192), O(8192,2048) and unpool(8192,4096). It caches the
actual eight input rows once per CTA, uses b128 weight loads, register
prefetch, WMMA and a fixed in-CTA K-split reduction. Device tiles return FP32;
the standalone wrapper rounds to BF16. Default shape choices use16 waves,
BN32/BK64 for QKVG/joint and BN64/BK128 for O/unpool. Idle CTAs skip input
loads. `bench_skinny_gemm.py` defaults to the selected configuration; --sweep
repeats the lever comparisons, and --all-shapes covers all three unique
shapes. It validates against FP64 and measures identical-byte streaming reads
and the native AITER projection with the same graph timing helper.
`check_skinny_gemm.py` checks two seeds and bitwise invariance across CTA
assignments1/37/192. The8-wave/BN16/BK256 configuration is rejected by the
wrapper after failing numerical validation in tuning.

S1 now launches one kernel on that core. Q/K pairs exchange projected BF16
values carried as FP32 words through scoped stores/loads and private128-byte
flag lines; V/G write directly. A single masked relaxed atomic increments each
Q/K producer's launch epoch, followed by a CTA-wide broadcast. Waits cap at4096
polls and read the cheap per-CU cycle counter every64 polls. There are no
whole-cache WB/INV fences, shared polled flags, second math launch or peers.

Create `make_workspace(device, ctas=...)` once before timing or graph capture,
and pass it to qkvg. The optional convenience allocation performs a one-time
state zero outside the math kernel. Workspace is single-stream and tied to its
CTA count; check its final72 status words after execution. A timeout reports
false for the affected head and the caller must discard that workspace. The
small-CTA fallback keeps both halves in one CTA, preserving bounded behavior.
The production runtime still does not dispatch this experimental stage.

`check_s1_replay.py` captures1000 changing-sign inputs for each layer mode,
with seeded per-CTA delay and device-side verification of Q/K/V/G plus touched
FP8 cache values on every launch. It checks exact epoch progression, then
skips one producer and confirms the stale flag is rejected and the consumer
returns within its poll bound. A separate skewed-counter microprobe compares
the earlier plain update to the masked atomic; that microprobe alone did not
reproduce the earlier full-fusion failure, so its precise cause is unproven.
The full suite and replay tests are the correctness evidence for the selected
atomic/broadcast protocol.

Current reports and logs are under coord/phase3b.0-core.md and coord/logs/phase3b.*.
The earlier two-launch timing is historical; current measurements are in the Phase3b report.


Phase3c S2 optimization is in `stages/attention.py`. One task owns a context
chunk for all four KV heads. Raw FP8 loads span contiguous256-byte NHD rows;
chunk page IDs are loaded once, followed by register double/triple prefetch.
The default is BN64/eight waves/two prefetched tiles. There are up to192
context splits. Merge uses256 consumers (one packed query/head row each),
spread over all assigned CTAs. Partial data is laid out by consumer then split
so merge reads are contiguous. Scoped b128 instructions are batched before
one wait; scalar LSE reads use a coalesced layout. An eight-CTA arrival tree
publishes private128-byte completion lines. No polled flag has multiple readers.

Reuse `make_workspace` only on serial launches on one stream and with its
configured CTA count. Device length must be8..context, which is at most131072;
active page/map entries must be valid. All256 status words must equal1. After
any timeout discard workspace and output. Every wait has4096-poll and100-million
shader-cycle bounds, with a clock check every64 polls. Allocation/zeroing is
outside graph capture; S2 itself is one launch.

`mapping_page_aligned=True` requires the caller to guarantee
`map[p*16+i] == map[p*16]+i` for active pages and all offsets. Ordinary paged
SWA extension preserves offsets; `alloc_extend_swa_tail` can shift them and
cannot assume this property. Arbitrary maps use prefetched token lookups.
The manual oracle cases retain the token-permuted compact-cache test; the
`--page-map` option adds page-preserving allocation cases.

Manual GPU0 commands, each requiring `timeout -s KILL 300`:

```bash
python3 test/manual/kernels/muse_spark_mono/check_s2.py --all
python3 test/manual/kernels/muse_spark_mono/check_s2.py --window 2047 --page-map
python3 test/manual/kernels/muse_spark_mono/bench_s2.py
python3 test/manual/kernels/muse_spark_mono/profile_s2.py --waves 8
python3 test/manual/kernels/muse_spark_mono/check_s2_replay.py
```

S2.opt2 validated the current revision:12 FP64/AITER cases (including the
arbitrary token-mapped compact sliding cache), plus200 delayed changing-input
captured replays, a non-page-aligned131067 device length, NaN cache tail, and
missing-CTA bounded failure. The missing CTA's own consumer retains its
sentinel; the other255 consumers fail. Targets30us full,8us sliding,4us handoff
remain unaccepted. See coord/phase3c-S2-opt2-report.md for current evidence;
coord/phase3c-S2-opt-report.md is the preceding optimization's stop report.
`profile_s2.py --before` reads the archived baseline instrumented source in
coord/logs; it calibrates realtime ticks against CUDA-event elapsed time.
Profiling perturbs timings and is excluded from reported end-to-end latency.


`ablate_s2.py` classifies the partial loop with local shader-cycle and realtime
boundaries: raw loads, synthetic register-only math, FP8/layout conversion,
and full, plus empty/opaque-touch controls. The full diagnostic matches the
production partial outputs and LSE bitwise. Conversion anchors expand BF16 to
FP32 for an opaque register use, and the touch control spills, so these are
qualitative ablations with different resource use, not additive latency terms.
Assembly and per-CTA raw samples are saved under coord/logs/phase3c-S2-opt2-*.

`gist.Dockerfile` builds the separate pinned Triton3.9/uTLX/AITER baseline.
The existing Triton3.8 development container is unchanged. AITER installation
must set `AITER_USE_SYSTEM_TRITON=1` to preserve the requested compiler.
`bench_s2_stack.py --prepare` saves one shared random input file; `--impl`
(stage/utlx/aiter) and `--tag` compare images on identical tensors and timing.
`repro_dense_partial.py --output-prefix PATH` retains the first-device-compute
protocol and separates the new-toolchain evidence from the old3.8 logs.


The derived image uses a `.pth` bootstrap from `gist_sitecustomize.py` to load
ROCm SDK libraries with the public `rtld_global=False` option before Torch;
this avoids collisions with Triton's exported LLVM symbols. Its SDK library
path is also supplied to AITER's JIT linker. Build from this manual directory
as context so the bootstrap file is available; exact commands are in coord/env.md.
On shared tensors, nativeuTLX3.9 measures68.26us full/19.25us sliding and passes
FP64. Dense-partial TP4/TP8 both pass on the new stack (sum error<8e-7);
the old3.8 failure does not reproduce there. Torch/ROCm also differ between
images, so this does not identify a unique compiler fix. S2 is frozen for Phase3; its TDM optimization is deferred to Phase4.


S2.opt3 keeps the full-context implementation and groups sliding work by KV
head: eight chunks per head,32 active producers/consumers,160 inactive CTAs.
Each consumer polls its own128-byte line and merges eight query/head rows;
payload uses b128 stores with one drain before the head arrival counter.
The factory accepts `sliding_splits=4` for the bounded comparison or0 for the
legacy path; grouped sliding requires eight waves. Full attention retains
its previous eight/16-wave support.

On the opt2 shared inputs, final medians are63.37us full and26.28us sliding
(previous62.15/33.69us). The45/15us opt3 targets remain unmet. Direct raw FP8
operand-layout loads pass the shared-input oracle but regress to240-481us;
that candidate is archived under coord/logs and is absent from production.
The final12-case FP64/AITER suite passes. Both full and token-mapped sliding
pass200 delayed changing-input replays, NaN tails, and bounded missing-arrival
checks in separate processes. Sliding's missing head has56 failed consumers
and8 absent sentinels, while the other192 consumers succeed.

```bash
python3 test/manual/kernels/muse_spark_mono/sweep_s2_opt3.py
python3 test/manual/kernels/muse_spark_mono/check_s2_replay.py --window 2047
```

See coord/phase3c-S2-opt3-report.md for before/after evidence; that gate is accepted.


Phase3d now includes S3 (`stages/attn_tail.py`), S4 (`stages/route.py`),
and S6 (`stages/tail.py`), each a device function plus one-launch wrapper,
using the accepted shared skinny core. S5 experts awaits a separate plan.
See coord/phase3d-report.md and coord/status.md for the review checkpoint.

The S3 failure was resolved by source-register XCNT waits in the inline VMEM
helpers. `test_scoped_xcnt.py` is a permanent16-wave/126-VGPR masked int32
store/readback regression:4,194,304 words, zero wrong after the fix; the old
store helper fails3,916,212 words. Existing store/load completion waits and
CTA joins still enforce payload publication. S2 changed only to receive the
VMEM correctness wait; no attention tuning occurred.

S3 uses four virtual ranks, each with a separate o_proj shard, in one GPU0
launch. It sums BF16 partials in ascending rank order, then applies the native
BF16 branch / FP32 stream boundary. Every consumer has a private128-byte flag
and readiness line. The approximately64us timing includes all four virtual ranks; it is
not a prediction of four-device latency. S3 forwards its logical task index
to the exchange; a17-CTA cyclic permutation preserves outputs bitwise.

S4 preserves FP32 high/low router projections, unbiased sigmoid weights,
bias-only selection and lower-id ties; its single slot-table owner matches
native BLOCK_M16 ordering and padding. Its16 cases include deliberate ties
and all four contiguous64-expert ranges. `check_route.py --prepare` saves
identical inputs; `--native-reference` runs in muse-mono-gist-opt2 (3.9), then
the default mode validates the stage in muse-mono-phase0 (3.8). The native3.8
joint GEMM calls the already-known broken `_dense_partial`; native3.9 passes
FP64 on those same saved tensors. Saved refs also cover negated inputs for
changing-input replay. This is a labeled cross-stack reference, not a compiler
performance attribution.

S6 preserves the unpool BF16 store, BF16 input-scale product, BF16 post-norm
output, FP32 residual stream and optional BF16 pre-norm input. Six cases cover
two seeds, normal/last-weighted/last-unweighted paths. Four repeated two-value
epilogue fixtures match native math bitwise and distinguish the pre-norm dtype
path. S6 inputs are independent from weight RNG draws. Matched-projection
output and stream are bitwise against SGLang for all six full cases.

```bash
python3 test/manual/kernels/muse_spark_mono/test_scoped_xcnt.py
python3 test/manual/kernels/muse_spark_mono/check_tail.py --stage S3 --replay
python3 test/manual/kernels/muse_spark_mono/check_route.py
python3 test/manual/kernels/muse_spark_mono/check_tail.py --stage S6 --replay
```

Use GPU0-only visibility and `timeout -s KILL 300` as above. S6 also has
`--prepare` in the3.8 container and `--native-reference` in the3.9 container;
its ordinary validation compares the3.8 unfused native sequence,3.9 fused
native sequence and FP64 on identical tensors. `--seed 20261117` repeats S3/S6.
Reference-container import dependencies added for this phase:orjson3.12.0,
pyzmq27.2.0; Torch/Triton/ROCm versions are unchanged.

Final stage medians (seed7): S3 63.46us, S4 70.31us, S6 normal35.31us. S4/S6 remain
slower than their native references; the report includes both native stacks,
read comparisons, all errors and reproducible log paths. Timings use100 graph
samples and weight working sets at least16x the4MiB L2. Delayed replays check
400 S3 launches,200 S4 launches (six outputs each),1200 S6 launches; each stage
also rejects missing arrivals and stale tags. Fault tests run last in their
process, and workspaces are discarded after failures.

Stop at `READY FOR REVIEW: phase 3d`. No S5, persistent-layer wiring, or physical
peer-memory work is part of this checkpoint.

## One-rank layer composition (phase 4.1–4.3)

`check_layer.py` composes S1–S6 as device functions in one persistent rank
launch, with the two collective peers supplied by GPU-local stand-ins. Run with
`--all-ranks --context 131072 --left -1 --seed 7 --replays 200`; sliding uses
`--left 2047`, and the other fixture seed is `20261117`. The same GPU0 visibility
and `timeout -s KILL 300` requirements above apply. `--missing` drops one
producer at each stage in a fresh workspace. Any failed launch invalidates the
whole step, even if independent KV heads have already written their slots.
Discard the workspace after a failed launch.

CTA task ownership is `task % grid`, with increasing task indices. Last-arriver
merges use fixed split summation order. Each polled consumer owns a 128-byte flag
line. S2 consumes a complete Q/K/V group; S3 consumes that group's attention
rows. Subsequent M8 projections require all eight input rows, and slot-table
construction requires all eight routing rows. These are data joins, rather than
all-CTA barriers. Idle waits perform normal cached reads of future dense weights;
expert weights are not read before routing. Resource admission checks the actual
specialized kernel's registers, LDS and HIP occupancy before its first launch.

`isolate_layer_route.py` reproduces the original full8k rank1 order change and
compares integrated and standalone S4 against native and FP64 on identical X4.
Exact order remains required on identical S4 input. The composed layer allows
upstream rounding to reorder the selected set, compares scores by expert ID,
and compares terminal values. Let delta be the maximum absolute perturbation of
all 256 biased sigmoid scores in a row, computed in FP64 from the candidate's
and reference's X4. A changed top8 set is accepted only when the reference
8th/9th margin and every replacement's score gap are at most `2 * delta`.
Every row records delta, threshold, cutoff margin and both ID lists.

KV bytes outside the eight new physical (or mapped SWA) slots must remain exact.
At a new slot, a differing byte must have the same sign and an adjacent FP8 code.
Its recorded BF16 K/V divided by the descale must reproduce that code exactly.
Let delta be the candidate/reference pre-quantization value difference. The
reference value must lie within `2 * delta` of the midpoint of the two decoded
FP8 values. Every exception records the slot/head/dimension, both codes, both
values, midpoint, margin and bound. The standalone S1 suite retains its exact
cache-byte assertions. This policy follows the phase4.2 review; it does not
change kernel arithmetic or cache writes.

`--timing` captures peer preparation plus external event nodes bracketing only
the layer launch, then reports 100 untraced samples with prefetch off/on.
`bench_layer_stages.py` measures same-fixture standalone and native stage sums.
Native dense projections use unfused Torch BF16 linear plus native epilogues,
never the known-bad `_dense_partial`. Its collective comparison uses GPU-local
ordered FP32 additions; it does not measure physical GPU peer communication.
`profile_layer.py --left {-1,2047}` records every CTA's stage spans, private
wait events, and S5 task/wait detail. Instrumented snapshots include realtime
clock overhead; untraced graph measurements are authoritative. No auxiliary
capture buffer is present in this manual fixture.

An accepted cutoff replacement changes which expert contributes, so original
S5/S6 values are not an appropriate golden for that row. The checker saves the
original intermediates and computes native local S5 on the candidate's pool,
IDs and scores, checked against the FP64 expert oracle. It preserves the three
original stand-in peer partials, forms the ordered sum, and computes native/FP64
S6 using that row's candidate mid-stream. Only accepted replacement rows receive
these conditioned downstream golden values (relative max limit 0.02); other
rows retain the original composed reference. Every routing comparison is logged
before values, and value failures identify the tensor, rank, row, column and
per-row maximum. This follows the explicit rank1-X6 review answer; it changes
only the test reference, not the kernel, route selection or numerical tolerance
for the original layer path.

`bench_layer_native.py` adds the gist's fast native composition: AITER `tgemm`
BF16 dense projections and the unchanged uTLX paged-attention kernel. It retains
the native router and FP4 expert timings from the same fixture, and preserves
the earlier Torch-dense/AITER-attention composition as a labeled control. The
review's primary native sum uses the AITER/uTLX measurements. Native S3/S5 sums
include ordered GPU-local additions, so neither baseline measures real peer
transport. Expanded native SWA cache padding is zero initialized; active mapped
bytes remain identical to the compact fixture.

### Phase 4.5 measurement harnesses

Run only on GPU0 with both visibility variables set to `0`, inside the gist
Triton 3.9 image and under `timeout -s KILL 300`. These are measurements;
the accepted layer remains the default. All comparisons use `paired45.py`:
separate workspaces, alternating AB/BA, 200 samples per side (100 per order),
with raw samples retained under `/home/bangtliu/coord/logs/`.

- `bench_regular45.py --context {8192,131072} --left {-1,2047}` compares the
  actual dependent native layer graph against the monokernel, using the same
  stand-in collective primitive. Native SWA uses equivalent full physical pages.
- `bench_code45.py` compares runtime zero/full self-code reads in the same
  instrumented binary. Twelve idle CTAs replicate the ELF text six times,
  assuming CTA IDs map modulo six XCCs; placement is not instrumented. Every
  text checksum and layer output is checked. `--counter` imports no native
  or uTLX modules; the installed profiler could not produce usable counters.
- `bench_spatial45.py` measures one private split from `spatial45.py`: S1-S3
  on 512 CTAs, S4-S6 on 256. Host selection is scoped to this single-threaded
  harness; device math and dependency helpers match the accepted kernel.
- `bench_bandwidth45.py --bytes {75000000,400000000,1000000000}` compares a
  conventional grid, a resident grid, and that resident grid after a bounded
  barrier with one 128-byte flag line per consumer. Read checksums are exact.
- `bench_stages45.py` adds seven external event nodes around six native and
  standalone stage ranges. It reports effective minimum-byte throughput,
  including stage dispatch/event costs, rather than hardware DRAM traffic.
- `analyze_design45.py` is CPU-only and computes explicit source-based dominant
  weight/scale/KV budgets using the recorded route sets. It brackets TokenSpeed
  FP8/BF16 KV and distinguishes measured-node estimates from MI355 peak floors.

See `/home/bangtliu/coord/phase45-report.md` for findings and profiler limits.

### Phase 4.6 isolation

`bench_isolate46.py --mode overhead` compares the accepted default with the
measurement kernel at mask63. `--mode stages --stage {1..6}` compares one
runtime-selected stage inside that full binary with the identical stage body
compiled without the other five. `--mode barrier` compares runtime fine versus
grid-barrier mode in the same full binary, preserving intra-stage dependencies.
Every run uses GPU0, `timeout -s KILL 300`, the gist3.9 image, and `paired45.py`.

The measurement kernel is `layer46.py`; `generate_layer46.py` derives it from
`layer_accepted.py` using AST edits. Run the repository formatters after generation.
`Control[0]` is the runtime stage mask; `Control[1]` selects five grid barriers.
The barriers replace cross-stage fanout/waits only. They retain private128-byte
consumer lines, finite polls and admission of all768 CTAs at four CTAs/CU.
The workspace has12 status rows so barrier diagnostics cannot overrun it.
Epochs, counters and flags reset before timing. Skipped outputs come from the
accepted archive; selected outputs are poisoned before every replay.

`bench_existing46.py --stage {1..6}` supplies a second control: the existing
`common_*` standalone APIs on the same archived inputs, checking exact stage
outputs. These are distinct from compiling accepted bodies separately. In
particular, the accepted row boundaries request scoped I/O and packing while
the standalone boundaries use normal stores. Existing S5's FP32 reduced output
is checked after its BF16 cast outside the timed range. These ready-input tests
are not a dependent layer chain.

`analyze_spills46.py` attributes static scratch instruction sites through full
assembler inline source chains. It reads source snapshots under coord/logs,
because formatting can change debug line numbers after a measurement. Counts
are static sites, not runtime spill traffic. `phase46-text-equivalence.json`
checks executable text across formatting; each runtime-mask/barrier comparison
also checks complete HSACO identity. See `/home/bangtliu/coord/phase46-report.md`.

### Phase 4.7 chained baseline

`stages/chained.py` runs six dependent common-stage dispatches. S5 writes its
BF16 X6 boundary in its own dispatch; there is no extra conversion kernel.
Optional `failure` parameters preserve the old API paths when absent. With a
failure tensor, each dispatch checks the first failed stage and blocks later
stages. These are one-shot dispatch-boundary reads, not shared-line spin polls.
A failing stage lets all its CTAs finish the existing bounded protocol.

`check_chained47.py --seed {7,20261117} --context {8192,32768,131072}
--left {-1,2047} --all-ranks --replays 3 --missing` checks the archive bit for bit,
including complete cache hashes, delayed graph replays and a captured missing
producer test for every stage/rank. `bench_chained47.py --context {8192,131072}
--left {-1,2047}` compares the true dependent chain against `Regular` from
phase4.5 in alternating same-process graphs,200 samples per side/100 per order.
Use the gist3.9 image, both visibility variables0, and `timeout -s KILL 300`.
Source stage bodies remain unchanged; guards and the S5 output bridge live in
launch wrappers. The accepted one-launch layer remains the default.

The six-dispatch baseline passes all48 cases but misses the regular-graph
latency target; raw AB/BA differences are material. This is a working checkpoint
for the authorized pairwise fusion trials, not an adopted performance winner.

`bench_pair47.py --pair {12,34,56}` measures a five-dispatch candidate against
that six-dispatch baseline. These manual prototypes are generated by
`generate_pairs47.py` into `pairs47_device.py` plus its argument schema.
They preserve common-stage arithmetic and row layouts, use compiler-tracked
GFX1250 buffer stores at device scope for internal producer outputs, scoped
consumer loads, and a bounded barrier with128-byte private ready/decision lines.
The last arrival reads failure metadata once and fans decisions out privately.
The runtime capture adapter is confined to this single-threaded manual process.

The retention rule is >2% reduction in both AB and BA medians with positive
paired-bootstrap95% lower improvement bounds. `--mode warm` compares runtime
zero/full code reads within one binary; only pair56 exceeds64KiB. Its positive
code-warm result was followed by `--mode final` against the six-dispatch baseline.
All three candidates lost; **none is selected by production or chained dispatch**.
The retained baseline is the six-dispatch chain. These are rejected experimental
artifacts for review/reproduction, not alternative public kernel APIs.

### Phase 4.8 diagnostics (GPU0 only)

The accepted baseline is the six-launch `stages/chained.py` layer. Fusion is
shelved for this configuration. Run these manual scripts under the standing
`HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0` and `timeout -s KILL 300` policy:

- `diagnose_parity48.py --context 8192 --left -1`: balanced AA/BB blocks,
  independent first epoch parity, actual epoch readback outside the timed range.
- `diagnose_guard48.py --candidate 1 --shared --runtime`: current status reload
  versus register-carried success, same binary/workspace/QKVG output addresses.
  Candidate `2` disables cross-dispatch fail-stop for diagnosis only.
- `diagnose_ranges48.py --context 8192 --left -1`: seven external event markers
  around S1..S6 inside each dependent graph, compared with the real regular path;
  native compiled-kernel configurations are recorded outside timing.

Repeat with full/sliding (`--left -1`/`2047`) and 8k/128k contexts. Each final
comparison has 400 samples per side, 100 per order/parity cell. Event ranges are
instrumented diagnostic measurements; their medians must not be added or treated
as uninstrumented layer latency. No stage tuning or guard change was selected:
the controlled guard deltas were below the 2% redesign threshold. Default guard
mode `0` retains status reload and fail-stop. Modes `1`, `2`, and runtime `-1`
are diagnostic controls; runtime requires a third failure word supplied by the
manual harness. The 48-case archive suite, 144 delayed replays and 288 missing
producer graphs pass with the default. See `coord/phase48-report.md` for results.

### Phase 4.9 independent stage budgets

`chained.make_workspace()` selects the policies below. The explicit
`stage_policy=("common", "common", "common")` retains the phase 4.8 control.
Selection requires a >2% chain median gain in both orders and positive paired
bootstrap lower bounds; unmeasured 32k contexts retain common S3/S5.

| Context | S3 | S5 | S6 | GPU dispatches |
| --- | --- | --- | --- | ---: |
| Full 8k | common | k9 | wide | 10 |
| Sliding 8k | wide | common | wide | 9 |
| Full 128k | wide | k9 | wide | 12 |
| Sliding 128k | common | k9 | wide | 10 |
| Either 32k | common | common | wide | 7 |

The wide projection uses 16x16x512 tiles, four TDM buffers, two waves and
133088 B LDS, preserving the fixed SK4 tree and rounded projection bits.
S3 retains the original norm/gate arithmetic and non-packed AR/boundary.
S5 splits W13/W2/norm/sum-AR into separate dispatches; the expert roots use
k=9 (1728 CTAs), 110/91 VGPR, zero spills, and measured occupancy 9/10.
The accepted async.cg W2 read path remains in use. Only the separate-launch
expert output stores change to normal compiler-tracked stores.

`tune_chain49.py --stage s6|s3|s5` reproduces chain-context stage comparisons.
`bench_chain49.py --context 8192 --left -1` compares the selected dependent
chain with the real regular graph using balanced order and epoch parity.
Final measured chain/regular ratios are 1.060/1.059/1.010/1.040 for full8k,
sliding8k, full128k and sliding128k. All 48 archive cases, 144 delayed replays
and 288 missing-producer graphs pass. Stage suites cover native/FP64 references,
exact old-stage bits, changing inputs/routing and all virtual ranks.

The packed boundary is known incorrect for the current S3 layout and is excluded
from selection. Phase 4.9 is accepted/closed; its investigation is backlogged in
`coord/backlog.md` and must not be resumed without a new plan. `debug_packed49.py` is the
separately authorized one-attempt fault diagnostic; an exclusive marker in
`coord/phase49-packed-debug.attempted` prevents repeat execution. Do not remove
that marker to rerun a failed GPU case. `generate_bounds49.py` derives its
payload-access guards, including scalar accesses and complete b128 packets.
See `coord/phase49-report.md` and `status.md` before any further GPU work.


Phase 4.10 experiments are not selected until the full archive matrix passes.
`tune_chain410.py` contains warm single-layer screening controls. Current review
requires cold measurements: `two_layer410.py` runs two dependent layers with
distinct fixture-seed weights, balanced AB/BA order and epoch parity. The second
layer uses fixed identical virtual-peer payloads in both controls; this measures
cache policies, rather than distributed multi-layer model accuracy. `ci410.py`
keeps each two-replay block intact in the paired bootstrap and requires a positive
95% lower delta in both orders. `check_candidate410.py --suite` runs the existing
S2 FP64/native/page-map, delayed-replay and missing-producer suite with the split
partial/merge candidate; without `--suite` it drives the archive layer checker.
No experiment changes real peer transport or enables the packed S3 boundary.


The Phase 4.10 selected chain uses the following per-context budgets (the
Phase 4.9 table above is the historical control):

| Context | S1 | S2 | S3 | S5 | S6 | Dispatches |
| --- | --- | --- | --- | --- | --- | --- |
| Full 8k | common | split | common | k9 | wide | 11 |
| Sliding 8k | wide | split | wide | k9 | wide | 14 |
| Full 128k | common | split | wide | k9 | wide | 13 |
| Sliding 128k | common | split | wide | k9 | wide | 13 |
| Either 32k | common | split | common | common | wide | 8 |

The wide S1 projection preserves the fixed SK8 tree and original head epilogue.
Split S2 preserves the original split count and merge order, and removes its
in-kernel all-grid handoff. Long chunks pipeline the next QK before current PV.
`attention_backend="common"` and `projection_backend="common"` retain explicit
controls; `stage_policy` continues to select S3/S5/S6. The same S5 k9 kernel is
selected in all four measured contexts. Untimed 32k S1/S3/S5 choices are preserved.

Both complete matrices pass exactly: 48 cases, 144 delayed replays and 288
missing-producer graphs each. S1/S2 stage suites cover native/FP64 references,
page maps, cache bytes, changing signs, delayed replays and missing producers.
`bench_rotating410.py` compares the true regular graph with three independent
rotating weight allocations, checking pointer uniqueness and native/archive
outputs. Final chain/regular ratios are 1.037/1.014/0.979/1.019 for
full8k/sliding8k/full128k/sliding128k. Only full128k beats regular in both orders;
the all-context performance target remains missed. See `coord/phase410-report.md`.

Unselected experiments are opt-in: `two_layer410.py --candidate` supports S1,
S3/S5 choices, expert non-temporal loads, loads before polls, and next-layer warm.
`--candidate combined --wide-s1-128` reproduces the rejected sliding128k stack.
`bench_register_projection410.py` checks three-copy WMMA-lane tiled weights with
direct b128 non-temporal operand loads; it loses for S1/S3/S6. `probe_cache410.py`
intentionally compares same-buffer reuse with three rotating arrays, without
claiming a MALL capacity. No tiled copies, warming, non-temporal expert loads,
changed-summation norms, packed S3 boundary or cross-layer fusion are selected.
The warm prototype uses 64 private consumer lines and waits until all eight S6
rows have consumed critical inputs. Every poll remains bounded.


T0 TLX feasibility uses the gist image's stock Triton3.9 plus `utlx_plugin`
(6c540ca), following `kernels/ops/attention/tlx/paged.py`. The alternate
`~/tlx_triton` compiler is not used. Production stages remain the Gluon
reference at `2d5315ec96`; these are manual feasibility proofs only.

- `inventory_tlx.py` runs without a GPU and records exported APIs, source
  revisions, and a mapping for all Gluon language symbols used by the stages.
- `proof_tlx_scoped.py` runs 1000 tagged two-CTA handoffs, a bounded missing
  producer, and the masked XCNT register-reuse regression against Gluon.
  It never executes a no-XCNT variant or a trap.
- `proof_tlx_projection.py` compares an S6-shape four-buffer TDM projection
  against Gluon's fixed SK4 tree, checks changing-sign replays and FP64 error,
  and measures three distinct rotating weight copies in balanced AB/BA order.
  A separate M16N64K128 dot proves WMMA code generation for the expert tile.
- `tlx_ir.py` saves TTIR, TTGIR, LLVM IR, AMDGCN and resources.
  `analyze_tlx_ir.py` generates literal diffs and static instruction inventories;
  column-zero inline-asm instructions are counted. These are not profiler
  counters. Each scoped VMEM asm block is checked for its XCNT wait.

GPU proofs require `HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0` and
`timeout -s KILL 300`; both refuse other device configurations. CPU-only tools
use an empty HIP visibility setting. Evidence is in `coord/phaseT0-report.md`,
`coord/phaseT0-api-map.md`, and `coord/logs/phaseT-ir/`. The final projection is
bitwise exact but about1.6% slower in this bounded proof; no stage is adopted.

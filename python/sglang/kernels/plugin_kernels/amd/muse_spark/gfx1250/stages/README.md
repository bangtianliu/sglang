# Common per-rank stage prototypes

The `common_*` modules expose the Phase 4 device stages and standalone validation
launchers: `common_qkvg` (S1), `common_attention` (S2), `common_attn_tail` (S3),
`common_route` (S4), `common_experts` (S5), and `common_tail` (S6). They are not
wired into the serving path or combined into a decoder-layer launch yet.

`framework.common_shape.CommonShape` carries four waves, k=4 CTAs per CU,
VGPR ceiling 224, and LDS ceiling 72 KiB. Grid and compiler options derive from
these parameters. Host admission checks actual compiled VGPR/LDS allocations
and HIP occupancy with Triton's dynamic LDS allocation included. Spill counts
are recorded separately; the attention compiler can spill within the VGPR cap.

S1/S3/S4/S6 use `streamed_gemm` with streamed X, asynchronous two-stage LDS
weight/input loads and external split-K partials. The split reduction preserves
the previous high-to-low bit order. S2 uses M64 attention tasks and FP8 LDS
staging. S4's `slot_table` has one wave own the 64 stable expert/slot keys,
with boundary ballots and padded prefix allocation; other waves do no slot
memory work. S5 retains the native-structure asynchronous expert core and uses
private per-expert-block flags for the W13-to-W2 handoff. The default W2
input path uses compiler-tracked async copies at device scope (`.cg`); explicit
benchmark controls retain invalidate-once/cached copies and batched scoped
register loads. The two-way W2 split-K experiment is optional and was slower
on the sparse fixture, so the default keeps the original accumulation.

Each S3/S5 launch computes exactly one rank's work. The test harness calls
`framework.standin.prepare` before the timed launch to seed the other three
reference partials and exact next-launch flags into GPU-local peer regions.
The chosen rank executes its own one-shot collective path, including puts,
private consumer flags, scoped loads and ascending-rank FP32 accumulation.
The helper never maps or accesses another physical GPU. Stand-in preparation
is excluded from timing with a separate setup graph.

Workspaces require serial use on one stream. Every poll is bounded and every
consumer has a private 128-byte flag line. A false status invalidates outputs;
discard that workspace after a failed launch. All stage tests run on GPU0 with
an external 300-second kill timeout. The `check_common_s*.py` manual suites
reuse saved hashed native references and FP64 oracles, changing delayed replays
and bounded missing-producer cases. `check_common_slots.py` adds duplicate-hit
stress and old/new slot-table timings.

S1/S3/S4/S6 use last-arriving split merges with the original fixed reduction
order. S1 runs the head epilogue in the merging CTA; its terminal diagnostic
markers contain the exact launch tag and need no competing clears. S3 folds
head norm/gate into the GEMM input tile. S3/S4/S6 release eight row owners on
private flag lines after the final N tile; S4 has a further eight-row join for
its slot table. S5 uses per-expert-block readiness, W2-owned hit norms and tagged
per-token completion counters, rather than stage-wide barriers. Its row owners
retain the fixed slot sum and one-shot peer sum. Independent tasks may finish
when a producer is missing: reject the entire launch if any required status or
exact completion tag fails, and discard the workspace.

S1/S3/S4 now meet their standalone latency targets, and S6 is near its target.
S5 still has a measured native-latency gap; layer overlap is not implemented.
No full-layer residency/performance claim follows from stage admission. Optional
private realtime traces record each CTA; the S5 task trace also separates waits
when a CTA processes several tasks. Tracing is compiled out of ordinary images.
Layer integration requires the subsequent review gate.


The single-rank layer experiment in `layer.py` defaults to the accepted phase4.2
kernel in `layer_accepted.py`. Its device function preserves the accepted source
exactly. Manual GPU0 experiments select a workspace variant explicitly:1 uses
producer squared-value statistics and larger apply fragments,2 uses per-row
last-arrival finalizers, and3 retains the earlier tiled-row diagnostic path.
Variants1/2 are selected by `TAIL_MODE` at compile time; neither beat the529us
acceptance threshold in the bounded phase4.4c pass. These are private benchmark
controls, with no serving integration or new multi-GPU authorization.


The phase4.4d packed-row epilogue is retained behind `PACKED_ROW=True` in
`boundary.norm_residual`; the default remains the literal accepted body.
The packed path removes standalone scalar vector loads and most spills, but
its full-layer timing reversed under interleaved control measurements, so it
is not selected. Manual4.4d checkers explicitly opt into the packed entry.

Layer performance claims require interleaved same-process A/B graphs with separate
workspaces, alternating order, at least100 samples per side and results for both
orders. The paired manual harness retains raw per-order samples; standalone
cross-process timings are diagnostic only (unchanged-layer spread529-615us).

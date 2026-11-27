# Supervised TP4/EP4 SPX bring-up — PREPARED, NEVER RUN

T12 preparation only. **Multi-GPU remains HALTED.** The host has crashed twice
in peer work. No result here establishes peer coherence, link health, bandwidth,
or four-GPU layer correctness. Do not rerun historical P.1. No partition change,
GPU reset, automatic retry, push, or PR. Have the user awake at the console and an
out-of-band way to observe/recover the host before proposing the first rung.

The source lives in `~/sglang-mono/test/manual/kernels/muse_spark_mono/bringup/`;
`~/coord/bringup/` links to these files. Original `~/sglang` stays on main.
The container image is `local/muse-mono-gfx1250:utlx-tuple-final` (stock Triton3.9,
patched uTLX); the source and fixture hashes are in `manifest.json`. Source drift
invalidates the bundle. These are unexecuted experimental adapters, not a serving
integration. Only CPU syntax/import and gfx1250 offline compilation were checked.

## Before each rung

1. Print the proposal, e.g. `python ~/coord/bringup/supervise.py rccl2`.
   This default path reads files and prints JSON; it never queries a GPU or starts
   Docker. Append the exact JSON as `PROPOSED RUN: T12.rccl2` to `status.md`.
2. Claude reviews code, command, expected memory traffic, and host health. Only
   Claude may create `~/coord/approvals/<sha256-of-file>.ok`: one for the bundle
   manifest and **each** Python file in this directory. No script creates these.
   `sha256sum ~/coord/bringup/manifest.json ~/coord/bringup/*.py` lists the hashes.
   Claude must also add the exact step-specific line to `review.md`:
   `APPROVED RUN: T12.rccl2 <sha256-of-manifest.json>`.
3. Set `MUSE_BRINGUP_APPROVAL_EXPIRES_UTC` to an explicit future UTC ISO timestamp
   with timezone (e.g. the end of the supervised slot). The worker and supervisor
   refuse missing, expired, or absent approvals **before any GPU interaction**.
   Then, and only then, the operator adds `--execute` to that one proposal.
   Approval does not authorize the next rung, another configuration, or a retry.
4. After the rung STOP. Inspect all rank logs, health snapshots and result JSON;
   Claude reviews before preparing/approving the next command. Revoke approvals
   after the supervised slot. Do not approve all rungs in advance.

|Order / step|Traffic and exact success condition|
|---|---|
|a1 `rccl2`|Physical GPUs0,1; RCCL FP32 all-reduce,1/32768/1048576 elements. All elements equal sum(rank+1).|
|a2 `rccl4`|Same on GPUs0..3. Every rank must report all3 exact sizes.|
|b `ping2`|GPUs0,1;128B uncached symmetric allocation/rank; one CTA each. Rank0 writes73 to rank1, rank1 polls and returns91. Both status=1 with observed tag and finite poll count. Uses our DEV-scoped helpers including XCNT/store drains. No bulk DMA.|
|c1 `oneshot2`|Two real ranks; new generic TLX adapter of our one-shot protocol.8 rows x4096/8192 BF16, three sequential epochs crossing parity. Scoped payload writes, private128B/source/consumer flags, scoped reads, ordered FP32 sums. Exact equality to RCCL for integer-valued BF16 inputs.|
|c2 `oneshot4`|Same with four real ranks. Saves TTGIR/LLVM/AMDGCN; refuses absent XCNT/DEV or cache-wide WB/INV. The adapter is not the production row reducer; d exercises that actual helper.|
|d `layer4`|Four physical ranks, accepted archived TP4/EP4 fixture (M8), all-TLX production S1..S6, actual mapped peer pointers for S3/S5. All4 contexts (full/sliding8k/128k). Exact transport/replay oracle plus native/FP64 checks; graph timing versus rank-aware true regular stages with RCCL collectives.|

## Watchdog, health and stop criteria

Each command starts a uniquely named disposable Docker container. The inner
`timeout -s KILL 60` bounds the complete torchrun (including imports/compilation,
collectives, checks and teardown); torchrun has zero restarts. The independent
host watchdog at65s kills that container and its launcher process group. It also
aborts on any plan/review mtime change. Gloo/RCCL operations have20s timeouts and
monitored control barriers10s. Polls are capped at65536, with sleep backoff and a
GET_REALTIME deadline checked every64 polls; these add an in-kernel bound. Layer
uses the accepted production4096-poll bound and stage failure/status records.

The wrapper captures `dmesg --ctime` and `amd-smi metric --gpu all --json` before
and after (5s command timeout). A failed before-snapshot refuses the run. Snapshots
are diagnostic GPU queries across the approved rung; none run during preparation.
All rank stdout/stderr are redirected under a timestamped
`~/coord/logs/bringup-<UTC>-<step>/`; launcher,proposal,result,IR and PASS records
share that directory. New driver/RAS/AER/fault/reset messages make the run fail.

STOP on timeout/hang, crash, missing PASS, negative/poll failure status, any
mismatch/nonfinite value, unexpected driver message, lost health snapshot,
protocol change, or an unresponsive host. Do not proceed or retry. Preserve logs
and report first tensor/rank/location, observed tag/polls and last completed event.
A watchdog cannot recover a wedged GPU/kernel driver or host; killing a process
is not proof that the node is healthy. Never auto-reset the device. A cold
compilation timeout is also a stop: review a CPU-precompile plan, not a longer
unreviewed GPU retry.

## Layer oracle and limits

The real layer workspace replaces S3/S5 scratch with `HipSymmetricBuffer`
uncached POSIX-FD VMM mappings and a table of actual per-rank mapped pointers.
No fabric-handle fallback, remote-host transport or partition change is attempted.
The base workspace flag strides are retained, including S5's original peer-stride
versus its wider compute grid. `prepare_peers` is **never** called on that real
workspace. All ranks start at epoch0, initialize only their local allocation,
and meet a CPU barrier before launch. This directly tests whether the existing
DEV-scoped protocol works across these physical mappings; it is not presumed.

The bitwise transport oracle runs isolated local TLX stages: extract each rank's
actual partial, exchange those tensors over CPU Gloo, seed virtual peers with
those actual values, then rerun the stage. This preserves local arithmetic and
rank0..3 sum order while independently checking physical communication. The
native/FP64 oracle separately checks the arithmetic under the accepted layer
numerical, routing-near-tie and FP8-midpoint contracts. It does not promise
bitwise equality to FP64. The adapter intentionally refuses unexplained errors;
it never applies the old virtual-only teacher-forcing rule to real ranks. A
near-cutoff replacement can require a reviewed real-rank reference extension.

The separate regular graph is a tracked rank-aware copy of `regular_layer45.py`:
AITER/native stages, correct local EP rank, and RCCL FP32 sums rounded to BF16.
Its Gluon definitions are reference-only; the TLX chain imports none of them.
Five AB and five BA trials per context and three one-shot epochs are smoke
numbers, not cold latency claims. Use >=3 rotating weight copies/two distinct
layers and two independent sessions before reporting a four-GPU headline.
This is one accepted seed and four contexts, not a production-serving matrix.
Full seed/rank/replay/missing-producer campaigns and server integration follow
only after this ladder is stable and separately approved.

# gfx1250 persistent framework

This package uses stock Triton3.8 Gluon. It imports nothing from `sglang.srt`.
During development every peer pointer addresses an allocation on GPU0; real
peer memory is unvalidated and requires separate exact-script approval.

`sync.publish` joins a CTA's payload stores, then releases a32-bit flag using
`atomic_xchg`. `sync.wait_flag` polls that flag with acquire semantics, sleeping
between retries, and returns the last value after either its iteration cap or
its realtime deadline. Callers must check equality before consuming data.
Agent scope (`gpu`) is the default. The explicit system-scope option is for
future reviewed peer mappings; it has not been validated across GPUs.

The gfx1250 realtime instruction is `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`,
followed by `s_wait_kmcnt`. HIP's `wall_clock64()` emits that same sequence.
A host-time measurement over one second observed99.88MHz; the deadline timebase
is100MHz. The old `s_memrealtime` builtin fails compilation on this target, and
the image's Gluon libdevice exposes no clock helper. Backoff is `s_sleep 1`.

All GPU invocations must expose only `HIP_VISIBLE_DEVICES=0` and use
`timeout -s KILL 300`. Polls always have an iteration cap as well as a deadline.
Trap handling is tested last, in its own process; compiling the `sync.trap`
helper does not imply its exception behavior has been validated.

The initial grid-barrier probe completed at both192 and193 CTAs on this192-CU
GPU (eight warps,64KiB explicit LDS plus32 bytes of compiler scratch per CTA).
Therefore `CU count + 1` is not an empirical non-residency bound for this kernel.
The192-CTA barrier had a median arrival-to-completion time of82.56us over1000
CUDA-graph replays. This does not authorize a larger production grid; the
one-workgroup-per-CU launch policy remains the conservative contract. A separate
missing-arrival test rejects a stale tag and returns false through the bounded
timeout path (1 ms configured deadline, 1.77 ms observed including polling
overshoot). The planner accepted the 193-CTA result.


Step 2.2b compares the all-to-all flag barrier with a single arrival counter and
with 16 groups of 12 CTAs feeding a top counter. Counter and flag words are
separated by 128 bytes. Counters start at zero, tags start at one and increase
monotonically; reinitialize storage before a tag wraps. These remain experimental:
the fastest measured variant takes 56.48 us median and misses the 5 us target.
Do not adopt a counter variant as the production barrier until reviewed.

The manual `bench_barrier.py` records all 192 CTAs over 1000 graph launches per
replay, with three replays for host timing. A runtime enable word disables the
barrier in the baseline while preserving the identical compiled kernel, register
count and live LDS allocation. A separate one-CTA launch increments the epoch;
its cost cancels in the host timing difference. Timings cover the whole CTA
arrival-to-release interval, including fences and CTA joins. Poll/sleep counts
and raw per-CTA medians accompany the results under the coordination log path.
All measured assembly uses agent scope, with no `SCOPE_SYS` instructions.


Step 2.2d adds experimental `scoped.py` for gfx1250 agent-coherent word
accesses. Every cross-CTA payload load and store must use these helpers: they
are not general acquire/release wrappers for ordinary cached memory accesses.
`store_words` issues masked `global_store_b32 scope:SCOPE_DEV`; every wave
waits with `s_wait_storecnt 0` and joins the CTA before one lane publishes the
flag. Consumers issue `global_load_b32 scope:SCOPE_DEV` and wait for completion
before reading the scoped payload. No whole-cache writeback or invalidation is
emitted. Masked assembly restores EXEC and declares the SCC clobber.

Counter variants drain every CTA's scoped stores before a relaxed agent atomic
arrival. Each returned counter ticket establishes completion of that arrival;
the last arrival publishes a scoped tag. A two-level variant uses separate
128-byte counter/flag lines. Flag polls load one physical lane and broadcast
through a CTA reduction. Both an iteration cap (4096 by default) and a realtime
check on every iteration bound all scoped polls. Missing-arrival tests exercise
the iteration bound and deadline independently, including stale tags. All 1001
launches of the 4 KiB payload test are checked individually through a results
array. No real peer memory is involved.

This protocol is still too slow: the final 192-CTA single-counter measurement
is 41.88 us median / 60.04 us max, with a 53.25 us host timing difference. The
5 us target is unmet; do not promote these experiments to the serving path.

The abandoned volatile experiment was bounded per launch, but the harness
queued 1000 launches per graph and three replays before checking results. A
single-launch reconstruction reaches its million-poll limit in about 49 ms,
so the queued workload can consume minutes. Its saved assembly contains the
iteration increment/comparison and repeated loads. A final acquire reread also
hid failed polls by replacing their observed value; `counter_barrier` now
preserves the success decision before that reread. The benchmark now checks an
enabled preflight before graph capture. This prevents a long queue of failing
launches; it does not replace the device bounds or the external timeout.


Steps 2.2f–2.2g supersede the earlier barrier performance blocker. Scoped flag
loads from one shared cache line cost 25.43 us per poll at 192 CTAs; giving each
CTA a separate 128-byte completion line reduces that to 0.48 us. The last
arriver writes every completion line. This is now the counter barrier's default;
allocate `(GRID // GROUP + 1 + GRID) * 32` int32 words, initialized to zero, and
use launch tags starting at one. Counters remain cumulative across launches.

The single-counter barrier with spread flags and realtime checks every 64 polls
measures 3.64 us median / 6.48 us max; the 16-groups-of-12 variant measures 2.92 us
median / 7.40 us max (192 CTAs, eight warps, 64 KiB live LDS, 1000 graph probes).
Every poll still checks its iteration limit. Deadlines can overshoot by up to
one 64-poll batch plus scheduling/memory latency: a missing-arrival test with a
1 ms deadline completes within 1.04 ms for spread flags. Tests independently
verify iteration-cap termination and deadline termination, then verify 1000
launches of 4 KiB per-CTA payloads read by neighboring CTAs after the barrier.

The image compiler maps `__builtin_readcyclecounter()` to
`s_get_shader_cycles_u64`; `sync.shader_cycles` exposes that diagnostic counter.
Its measured overhead is about one shader cycle per read. Deadlines keep using
the nominal 100 MHz realtime clock, so they do not depend on shader frequency.


`plan.make_plan` lays out named 32-bit word regions, padding every per-rank
region to 128 bytes. Cross-rank regions have two copies selected by the launch
tag's parity; local regions have one. `mailbox` provides local and peer forms
of scoped puts, tag publication and bounded waits. The peer base is always a
kernel argument or loaded pointer-table value; this framework maps no memory.
One consumer owns each flag line. The static three-stage pipeline sends all
producer data before waiting downstream, so it cannot form a cycle of waits.

`collective.all_reduce` parameterizes the virtual world size (four by default,
with four and eight tested). Each rank owns a contiguous `1 / world_size` of
the output and splits that shard among its CTAs. Each source pushes BF16
partials converted to FP32 words into the reducer's parity region, then raises
one flag per consuming CTA. Each reducer adds sources in the explicit order
`0 + rank0 + rank1 + ... + rank(world_size - 1)`, strictly left-associated,
without BF16 rounding. Reducers push completed FP32 shards to all ranks and fan
out separate completion flags. Consumers wait on their own lines and copy the
full gathered sum. Both phases partition work identically, including uneven
shards at 24 CTAs/rank. All producers issue their phase's puts before its waits.
Input tensors are immutable during each launch; cross-CTA scratch always uses
scoped accesses. Virtual peer pointers currently all refer to one GPU0 scratch
allocation; real peer visibility remains unvalidated.

The all-reduce probe checks both `[8,8192]` and `[8,4096]` per-rank BF16 partials
against left-to-right FP32 torch addition. Every one of 1000 validation launches
uses a different power-of-two scale in a seven-launch cycle, so reusing the same
parity slot's old payload is detectable. All ranks and elements are checked in
every launch. Performance uses a separate graph containing only the collective
and epoch increment, subtracting an epoch-only graph baseline.

`stuck.wait_or_record` writes a valid marker only after its diagnostic fields
have completed to system scope. Unlike normal mailboxes, a host-readable record
uses system-scoped stores so CPU readers can inspect it even after an isolated
trap process exits. The test pins a file-backed host page with local host-memory
registration, not a peer mapping. The diagnostic contains tag, polled address,
observed value, realtime counter, virtual rank, CTA id and numeric stage id.
A device-buffer/readback path is also tested. The trap is opt-in and invoked
only in the final isolated test after all ordinary GPU validation is complete.


The pipeline test also runs 100000 launches at each of world sizes four and
eight with deterministic per-CTA, per-launch delays of 0..127 `s_sleep` calls
before producer stores, between stores and publication, and before consumer
waits. CTA-to-stage roles rotate each launch. Both transform and consumer
comparisons retain the first mismatch record per rank/stage (launch, rank,
stage, index, expected, got). Timestamped JSON records are written before host
assertions, including a separate diagnostic for CPU-reference mismatches.
The earlier isolated host comparison failure remains unattributed; the timing
stress passes, including stale-tag injection. Step 2.7 switches the Phase 1
stage-0 seam to TP4/EP4; real peer execution and serving-weight validation
remain future work.


Step 2.8 selects `collective_fast.oneshot` as the preferred virtual all-reduce.
It pushes the unchanged BF16 bit patterns to every peer, then accumulates the
received sources in FP32 in rank order. A `[8,8192]` BF16 partial is 128 KiB;
packing two BF16 elements per word avoids FP32 expansion during communication.
The single hand-off round is faster here than reduce-scatter plus all-gather.
`collective_fast.batched_rs_ag` remains a tested comparison; the earlier scalar
`collective.all_reduce` retains optional six-phase timing for the baseline.

`vector_memory` issues agent-scoped b128 accesses with explicit register-tuple
constraints. A source batch issues all masked loads before one load-counter
wait. A destination batch keeps every packet's fixed input registers live until
one store-counter wait completes inside that same assembly block; only then
does the CTA join and publish all destination flags. This lifetime guarantee
is necessary: the separate-store experiment failed the permanent packet-copy
regression. The final tests cover six distinct source/packing/wave configurations,
including masked tails and different packets for every destination. Explicit
register batches stay below v128; the largest world-eight packet configuration
uses sixteen waves to stay in the verified batch range. Flags are polled in
parallel by source lanes, still with one 128-byte line per consumer, an iteration
cap on every poll, and a realtime deadline check every 64 polls.

At world four, the preferred path measures 3.04 us median / 8.72 us max for
`[8,8192]` at 32 CTAs/rank and 2.88 us median / 7.92 us max for `[8,4096]` at
16 CTAs/rank. These are per-CTA in-kernel intervals with two realtime reads;
host graph differences are reported separately. The full matrix retains both
algorithms at worlds four/eight, both widths, and the tested CTA counts. Every
case checks all ranks/elements in 1000 changing launches. An explicit BF16
cancellation fixture makes reassociation change the answer, guarding the fixed
FP32 summation order. The manual probes emit six-phase JSON measurements and generated assembly
for review.

### gfx1250 inline VMEM source-register lifetime (XCNT)

LLVM's [gfx1250 XCNT change](https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20250623/1665084.html)
tracks VMEM/SMEM source register uses and waits before redefining them. For VMEM
this includes store data/address VGPRs and load address VGPRs. Inline asm hides
these instructions from compiler hazard tracking, so every VMEM block here
ends with `s_wait_xcnt 0x0`. Loads also retain `s_wait_loadcnt 0` for returned
values; stores retain explicit store-counter drains and CTA joins before
publication. XCNT protects register reuse, not payload publication ordering.

Audit: `scoped.store_words`, `load_words`, `load_flag`, `load_words_masked`;
`vector_memory._store_asm` / `_load_asm`; attention's `_vector_asm` store/load
branches. Mailbox, collective, QKVG, attention-tail and boundary functions call
those helpers or compiler-tracked Gluon memory operations. They contain no
additional raw VMEM. Sync's clock/sleep/trap asm contains no VMEM. Compiler-
tracked atomics need no manual inline-asm XCNT addition. Existing b128 batch
store waits stay for completion; the explicit XCNT wait owns source-register
reuse safety even if batching or register allocation changes.

`test/manual/kernels/muse_spark_mono/test_scoped_xcnt.py` checks distinct words
under16-wave register pressure, source reuse between stores, masked lanes,
and exact scoped readback. This is a permanent GPU0 manual regression; its
pre-fix control and before/after results are in coord/logs/phase3d-S3-fix-*.

"""Layer dependency plan, private bounded waits and cached idle weight reads."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped, sync

# Flag offsets are128-byte lines, including padding for the largest full binary.
S2_INPUT = 0
S2_MERGE = 512
S3_INPUT = 768
S3_ROWS = 1280
S4_INPUT = 1288
S4_ROWS = 1864
SORT = 1872
S5_INPUT = 1873
S6_INPUT = 2641
S6_ROWS = 3153
S1_HEAD = 3161
FLAG_LINES = S1_HEAD + 288
STAGES = (
    "QKVG",
    "attention",
    "attention_tail",
    "routing",
    "experts",
    "unpool_boundary",
)


@g.jit
def warm(Base, N: gl.constexpr, pid, step, GRID: gl.constexpr):
    if N > 0:
        first = (step * GRID + pid) * 128
        i = first + gl.arange(0, 128, layout=gl.BlockedLayout([1], [32], [4], [0]))
        value = gl.load(
            Base.to(gl.pointer_type(gl.int32)) + i, i < N, 0, cache_modifier=".ca"
        )
        gl.inline_asm_elementwise(
            "v_mov_b32 $0, $1", "=v,v", [value], dtype=gl.int32, is_pure=False, pack=1
        )
        return gl.minimum(128, gl.maximum(0, N - first)) * 4
    else:
        return gl.full((), 0, gl.int32, gl.BlockedLayout([1], [32], [4], [0]))


@g.jit
def note(Diag, pid, stage, producer, task, tag, seen, polls):
    if gl.load(Diag + pid * 16) == 0:
        gl.store(Diag + pid * 16, stage)
        gl.store(Diag + pid * 16 + 1, producer)
        gl.store(Diag + pid * 16 + 2, task)
        gl.store(Diag + pid * 16 + 3, tag)
        gl.store(Diag + pid * 16 + 4, seen)
        gl.store(Diag + pid * 16 + 5, polls)
        gl.store(Diag + pid * 16 + 6, sync.clock())


@g.jit
def wait(
    Flag,
    tag,
    pid,
    task,
    Diag,
    Metrics,
    A,
    B,
    NA: gl.constexpr,
    NB: gl.constexpr,
    STAGE: gl.constexpr,
    PRODUCER: gl.constexpr,
    GRID: gl.constexpr,
    PREFETCH: gl.constexpr,
    WaitTrace=None,
):
    start = sync.clock()
    seen = scoped.load_flag(Flag)
    polls = 1
    elapsed = start - start
    step = gl.load(Metrics + pid * 32 + 8 + STAGE)
    count = gl.full((), 0, gl.int64, gl.BlockedLayout([1], [32], [4], [0]))
    while (seen != tag) & (polls < 1_048_576) & (elapsed < 10_000_000):
        if PREFETCH:
            if step % 2 == 0:
                count += warm(A, NA, pid, step // 2, GRID)
            else:
                count += warm(B, NB, pid, step // 2, GRID)
            step += 1
        seen = scoped.load_flag(Flag)
        polls += 1
        if polls % 64 == 0:
            elapsed = sync.clock() - start
    gl.barrier()
    gl.store(Metrics + pid * 32 + 8 + STAGE, step)
    gl.store(Metrics + pid * 32 + STAGE, gl.load(Metrics + pid * 32 + STAGE) + count)
    if WaitTrace is not None:
        gl.store(
            Metrics + pid * 32 + 21 + STAGE,
            gl.load(Metrics + pid * 32 + 21 + STAGE) + polls,
        )
        count = gl.load(Metrics + pid * 32 + 31).to(gl.int32)
        # Three bounded 21-bit poll counts per word; legacy wait records stay intact.
        if count < 12:
            word = count // 3
            at = gl.where(word < 2, 7 + word, 26 + word)
            old_polls = gl.load(Metrics + pid * 32 + at)
            gl.store(
                Metrics + pid * 32 + at,
                old_polls | (polls.to(gl.int64) << ((count % 3) * 21)),
            )
        if count < 16:
            base = WaitTrace + pid * 64 + count * 4
            gl.store(base, STAGE)
            gl.store(base + 1, start)
            gl.store(base + 2, sync.clock())
            gl.store(base + 3, task)
        gl.store(Metrics + pid * 32 + 31, count + 1)
    if seen != tag:
        note(Diag, pid, STAGE, PRODUCER, task, tag, seen, polls)
    return seen == tag


@g.jit
def arrive(Counter, tag, N: gl.constexpr):
    scoped.drain_stores()
    ticket = gl.atomic_add(Counter, 1, sem="relaxed", scope="gpu")
    return ticket + 1 == tag * N

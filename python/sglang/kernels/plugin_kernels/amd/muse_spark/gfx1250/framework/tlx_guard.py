"""Dispatch guard and deterministic bounded validation delays in Triton tl."""

import triton
import triton.language as tl


@triton.jit
def enabled(Failure, STAGE: tl.constexpr):
    if Failure is None:
        return tl.full((), True, tl.int1)
    else:
        failed = tl.load(Failure)
        return (failed == 0) | (failed == STAGE)


@triton.jit
def finish(Failure, success, pid, STAGE: tl.constexpr):
    if Failure is not None:
        if not success:
            old = tl.atomic_cas(Failure, 0, STAGE, sem="relaxed", scope="gpu")
            if old == 0:
                tl.store(Failure + 1, pid)


@triton.jit
def delay(Seed, task, POINT: tl.constexpr):
    if Seed is not None:
        seed = tl.load(Seed)
        count = ((task * 1103515245 + seed * (POINT + 1)) ^ (task >> 3)) & 31
        for _ in range(count):
            tl.inline_asm_elementwise(
                "s_sleep 1\nv_mov_b32 $0,0",
                constraints="=v",
                args=[],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )


@triton.jit
def step_reset(Failure, Count, K: tl.constexpr):
    """Start of a step (one CTA, in the captured graph): count and clear a workspace's failure
    word. Count[0] = failed workspace-steps so far, Count[1 + K] = this workspace's last failed
    stage. Without it a single failed exchange disables the workspace for good: its later
    stages never publish again, so every rank times out at them in every later step."""
    failed = tl.load(Failure)
    if failed != 0:
        tl.atomic_add(Count, 1, sem="relaxed", scope="gpu")
        tl.store(Count + 1 + K, failed)
        tl.store(Failure, 0)

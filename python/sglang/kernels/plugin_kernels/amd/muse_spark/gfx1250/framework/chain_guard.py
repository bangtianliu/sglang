"""Dispatch failure checks and diagnostic register/runtime controls.

The accepted mode reloads status. Runtime mode is selected from Failure[2] by
the manual harness; mode 2 disables fail-stop and must remain diagnostic only.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def enabled(Failure, STAGE: gl.constexpr, MODE: gl.constexpr = 0):
    if Failure is None or MODE == 2:
        return gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    else:
        failed = gl.load(Failure)
        # Let all CTAs of the failing stage finish its bounded internal protocol.
        return (failed == 0) | (failed == STAGE)


@g.jit
def finish(Failure, Status, pid, STAGE: gl.constexpr):
    if Failure is not None:
        if gl.load(Status + pid) != 1:
            previous = gl.atomic_cas(Failure, 0, STAGE, sem="relaxed", scope="gpu")
            if previous == 0:
                gl.store(Failure + 1, pid)


@g.jit
def finish_value(Failure, success, pid, STAGE: gl.constexpr):
    if Failure is not None:
        if not success:
            previous = gl.atomic_cas(Failure, 0, STAGE, sem="relaxed", scope="gpu")
            if previous == 0:
                gl.store(Failure + 1, pid)


@g.jit
def enabled_runtime(Failure, STAGE: gl.constexpr, mode):
    if Failure is None:
        return gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    else:
        if mode == 2:
            allowed = gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
        else:
            failed = gl.load(Failure)
            allowed = (failed == 0) | (failed == STAGE)
        return allowed

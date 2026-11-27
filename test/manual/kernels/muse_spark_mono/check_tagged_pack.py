"""CPU regression for the T13 faults: pack4 padded a one-element hint poll.

The old 128-element/four-wave poll compiled with three undef pointers, tags and
masks per thread. Its handwritten loop read those padded operands. Reject that
shape at compile time; the 512-element hint keeps all pack operands defined.
"""

import argparse
import importlib.util
import sys

import triton
import triton.language as tl
import utlx_plugin as tlx
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_tagged as mailbox,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
)


@triton.jit
def hints(Pairs, Epoch, Failure, Record, Out, N: tl.constexpr):
    lane = tlx.require_layout(tl.arange(0, N), ROW1)
    tag = tl.load(Epoch).to(tl.uint32)
    index = (lane % 8) * 32 + 31
    rec = Record + ((lane // 32) % 4) * 8
    word, retries = mailbox.poll(Pairs + index, tag, lane < 8, Failure, rec, index)
    tl.store(Out + lane, word)


def compile_case(n):
    return triton.compile(
        ASTSource(
            hints,
            signature=dict(
                Pairs="*i64", Epoch="*i32", Failure="*i32", Record="*i32", Out="*i32"
            ),
            constexprs={"N": n},
        ),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": 4, "enable_fp_fusion": False},
    )


def main():
    global mailbox
    p = argparse.ArgumentParser()
    p.add_argument("--old-module")
    a = p.parse_args()
    if a.old_module:
        spec = importlib.util.spec_from_file_location("tagged_pre_fix", a.old_module)
        mailbox = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mailbox
        spec.loader.exec_module(mailbox)
    try:
        bad = compile_case(128)
    except CompilationError as error:
        chain = []
        current = error
        while current is not None:
            chain.append(str(current))
            current = current.__cause__
        assert any("pack4 poll" in text for text in chain), error
    else:
        calls = [
            line
            for line in bad.asm["llir"].splitlines()
            if "asm sideeffect" in line and "global_load_b64" in line
        ]
        assert any("undef" in line.rsplit('"(', 1)[-1] for line in calls)
        raise AssertionError(
            "old 128-hint poll accepted undef operands; must reject at compile time"
        )
    good = compile_case(512)
    calls = [
        line
        for line in good.asm["llir"].splitlines()
        if "asm sideeffect" in line and "global_load_b64" in line
    ]
    assert calls and all("undef" not in line.rsplit('"(', 1)[-1] for line in calls)
    assert (
        "global_wb" not in good.asm["amdgcn"] and "global_inv" not in good.asm["amdgcn"]
    )
    print("PASS: reject128, accept512; zero undef poll inputs, CPU only")


if __name__ == "__main__":
    main()

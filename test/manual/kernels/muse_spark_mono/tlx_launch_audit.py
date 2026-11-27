"""Audit kernels launched while the production TLX entry/setup is active."""

import hashlib
import json
import re
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from tlx_ir import dump
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain

PREFIX = "sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250."


@contextmanager
def audited_tlx(label):
    match = re.search(r"(?:phase)?(T[0-9]+)", label)
    phase_tag = match.group(1) if match else "T5"
    active = [False]
    records = {}
    ir_seen = set()
    original = CompiledKernel.launch_metadata

    def marked(fn):
        def call(*aa, **kw):
            prior = active[0]
            active[0] = True
            try:
                return fn(*aa, **kw)
            finally:
                active[0] = prior

        return call

    def metadata(kernel, *aa, **kw):
        result = original(kernel, *aa, **kw)
        if active[0]:
            fn = kernel.src.fn
            owner = fn.fn.__module__
            kind = type(fn).__module__ + "." + type(fn).__name__
            assert "gluon" not in kind, (owner, kind)
            assert owner.startswith(PREFIX) and owner.rsplit(".", 1)[-1].startswith(
                "tlx_"
            ), (owner, kind)

            def const(name):
                if name not in fn.arg_names:
                    return None
                index = fn.arg_names.index(name)
                return kernel.src.constants.get(
                    (index,), kernel.src.constants.get(name)
                )

            phase = owner.rsplit(".", 1)[-1] + "." + fn.fn.__name__
            if fn.fn.__name__ == "project":
                phase += "-S" + str(const("STAGE"))
            if fn.fn.__name__ == "expert_phase":
                phase += "-w13" if const("FIRST") else "-w2"
            digest = hashlib.sha256(kernel.asm["hsaco"]).hexdigest()
            records[digest] = dict(
                owner=owner, function=fn.fn.__name__, jit_type=kind, phase=phase
            )
            if phase not in ir_seen:
                dump(kernel, phase_tag + "-" + label + "/" + phase, "tlx")
                ir_seen.add(phase)
        return result

    failed = True
    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(CompiledKernel, "launch_metadata", metadata)
            )
            for name in ("launch", "prepare_peers"):
                stack.enter_context(
                    patch.object(tlx_chain, name, marked(getattr(tlx_chain, name)))
                )
            yield records
        failed = False
    finally:
        path = Path("/home/bangtliu/coord/logs") / (
            "phase" + phase_tag + "-launch-audit-" + label + ".json"
        )
        path.write_text(
            json.dumps(
                dict(
                    label=label,
                    failed=failed,
                    kernels=list(records.values()),
                    kernel_hashes=list(records),
                    all_tlx=True if records else None,
                    argv=__import__("sys").argv,
                ),
                indent=2,
            )
            + "\n"
        )

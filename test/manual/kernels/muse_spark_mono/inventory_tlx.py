"""CPU-only inventory of the installed uTLX plugin and compiler provenance."""

import inspect
import json
import re
import subprocess
from pathlib import Path

import triton
import triton.language as tl
import utlx_plugin as tlx


def main():
    exports = {}
    for name in dir(tlx):
        if name.startswith("_"):
            continue
        obj = getattr(tlx, name)
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = None
        exports[name] = dict(
            type=type(obj).__name__,
            module=getattr(obj, "__module__", None),
            signature=sig,
            doc=inspect.getdoc(obj),
        )
    repos = {}
    for name, path in [
        ("triton", "/root/Workplace/projects/triton-openai"),
        ("utlx", "/opt/muse-gist/triton-ext"),
        ("alternate_tlx_fork", "/home/bangtliu/tlx_triton"),
    ]:
        repos[name] = dict(
            path=path,
            head=subprocess.check_output(
                [
                    "git",
                    "-c",
                    "safe.directory=" + path,
                    "-C",
                    path,
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip(),
            dirty=subprocess.check_output(
                [
                    "git",
                    "-c",
                    "safe.directory=" + path,
                    "-C",
                    path,
                    "status",
                    "--short",
                ],
                text=True,
            ).strip(),
        )
    source = Path(
        "/home/bangtliu/sglang-mono/python/sglang/kernels/plugin_kernels/amd/muse_spark/gfx1250"
    )
    used = sorted(
        set(
            name
            for folder in ("framework", "stages")
            for path in (source / folder).glob("*.py")
            for name in re.findall(r"\bgl\.([A-Za-z_][A-Za-z_0-9]*)", path.read_text())
        )
    )
    special = {
        "BlockedLayout": "MISSING direct public constructor; compiler layout inference",
        "DistributedLinearLayout": "MISSING direct public constructor",
        "DotOperandLayout": "tlx.dot_operand_layout; public AMD parent MFMA only, not WMMA",
        "SliceLayout": "tlx.slice_layout with supported parent",
        "PaddedSharedLayout": "tlx.padded_shared_layout_encoding",
        "SwizzledSharedLayout": "tlx.swizzled_shared_layout_encoding",
        "allocate_shared_memory": "tlx.local_alloc/view/reinterpret",
        "convert_layout": "tlx.require_layout/release_layout; public parent limitations apply",
        "to_linear_layout": "MISSING direct public encoding-query equivalent",
        "barrier": "tlx.workgroup_barrier (LDS only), or tl.debug_barrier (stronger waits)",
        "num_warps": "tlx.num_warps",
        "full_like": "tl.full(src.shape, value, src.dtype)",
    }
    constructs = {
        name: special.get(
            name, "tl." + name if hasattr(tl, name) else "MISSING/unmapped"
        )
        for name in used
    }
    result = dict(
        gluon_language_constructs=constructs,
        triton_version=triton.__version__,
        triton_path=triton.__file__,
        plugin_path=tlx.__file__,
        supports_multi_cta=tlx.SUPPORTS_MULTI_CTA,
        exports=exports,
        repositories=repos,
        tl_apis={
            n: str(inspect.signature(getattr(tl, n)))
            for n in (
                "dot",
                "load",
                "store",
                "atomic_add",
                "atomic_cas",
                "inline_asm_elementwise",
                "program_id",
                "num_programs",
                "range",
            )
        },
    )
    Path("/home/bangtliu/coord/logs/phaseT0-utlx-api.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                repositories=repos,
                exports=len(exports),
                supports_multi_cta=tlx.SUPPORTS_MULTI_CTA,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

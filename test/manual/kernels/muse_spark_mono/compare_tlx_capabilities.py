"""CPU-only side-by-side capabilities of the selected Gluon/uTLX installation."""

import inspect
import json
from pathlib import Path

import utlx_plugin as u
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import amd
from triton.experimental.gluon.language.amd import gfx1250


def describe(obj):
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = None
    return dict(
        signature=sig, doc=inspect.getdoc(obj), module=getattr(obj, "__module__", None)
    )


def main():
    pairs = {
        "descriptors": (gfx1250.tdm.make_tensor_descriptor, u.make_tensor_descriptor),
        "TDM_gather": (gfx1250.tdm.async_gather, u.tdm_async_gather),
        "TDM_scatter": (gfx1250.tdm.async_scatter, u.tdm_async_scatter),
        "TDM_load": (gfx1250.tdm.async_load, u.tdm_async_load),
        "TDM_wait": (gfx1250.tdm.async_wait, u.tdm_async_wait),
        "padded_LDS": (gl.PaddedSharedLayout, u.padded_shared_layout_encoding),
        "shared_linear": (gl.SharedLinearLayout, u.shared_linear_layout_encoding),
        "barriers": (gl.barrier, u.workgroup_barrier),
        "wave_partitioning": (gl.warp_specialize, u.async_task),
    }
    result = dict(
        pairs={
            n: {"gluon": describe(a), "utlx": describe(b)}
            for n, (a, b) in pairs.items()
        },
        amd_gluon_exports=[n for n in dir(amd) if "warp" in n or "pipeline" in n],
        utlx_exports=[
            n for n in dir(u) if any(s in n for s in ("warp", "pipeline", "async_task"))
        ],
        utlx_multi_cta=u.SUPPORTS_MULTI_CTA,
        scope="API/source comparison, not proof that every export lowers on gfx1250",
    )
    Path("/home/bangtliu/coord/logs/phaseT0-capabilities-addon.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                paired_features=len(pairs),
                amd_gluon_exports=result["amd_gluon_exports"],
                utlx_multi_cta=u.SUPPORTS_MULTI_CTA,
            )
        )
    )


if __name__ == "__main__":
    main()

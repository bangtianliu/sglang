"""Marked W13/W2 decomposition; test-only host launch proxies, no kernel edits."""

import sys

import bench_tlx_variant as bench
from paired45 import distribution
from tlx_ir import dump

original = bench.balanced


def measured(cases, epochs, observer):
    def observe(label):
        result = observer(label)
        mod = sys.modules[
            "tlx_custom_candidate" if label == "candidate" else "tlx_T6_baseline_5"
        ]
        for phase in ("w13", "w2"):
            result[phase + "_us"] = sum(
                a.elapsed_time(b) * 1000
                for a, b in [row[phase] for row in mod.PHASE_EVENTS]
            ) / len(mod.PHASE_EVENTS)
        return result

    result = original(cases, epochs, observe)
    result["component_timings"] = {
        label: {
            phase: {
                order: distribution(
                    [
                        r[phase + "_us"]
                        for r in result["raw"]
                        if r["label"] == label and r["order"] == order
                    ]
                )
                for order in ("AB", "BA")
            }
            for phase in ("w13", "w2")
        }
        for label in cases
    }
    return result


if __name__ == "__main__":
    ir_label = sys.argv[sys.argv.index("--label") + 1]
    bench.balanced = measured
    bench.main()
    for name, side in [
        ("tlx_custom_candidate", "candidate"),
        ("tlx_T6_baseline_5", "baseline"),
    ]:
        for phase, k in sys.modules[name]._PROFILE.kernels.items():
            dump(k, ir_label + "/" + phase, side)

"""Summarize cold stage A/B records with conditional paired-round bootstrap CIs."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="phaseT11")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    root = Path("/home/bangtliu/coord/logs")
    rng = np.random.default_rng(20261118)
    out = {}
    for path in sorted(root.glob(args.label + "-*.json")):
        d = json.loads(path.read_text())
        if "stage_ratios" not in d:
            continue
        raw = d["raw"]
        assert len(raw) % 8 == 0
        blocks = []
        for start in range(0, len(raw), 8):
            cells = {
                (r["label"], r["order"], r["parity"]): r["stage_us"]
                for r in raw[start : start + 8]
            }
            assert len(cells) == 8
            blocks.append(
                [
                    [[cells[(n, o, p)] for p in (0, 1)] for o in ("AB", "BA")]
                    for n in ("baseline", "candidate")
                ]
            )
        data = np.asarray(blocks)
        med = np.median(data, axis=(0, 3))
        point = med[1] / med[0]
        assert np.allclose(
            point, [d["stage_ratios"][o] for o in ("AB", "BA")], rtol=1e-12
        )
        indices = rng.integers(0, len(data), (args.draws, len(data)))
        boot = np.median(data[indices], axis=(1, 4))
        ratios = boot[:, 1, :] / boot[:, 0, :]
        out[path.stem] = dict(
            stage=d["stage"],
            context=d["context"],
            left=d["left"],
            ratios=d["stage_ratios"],
            ci95={
                o: np.quantile(ratios[:, i], [0.025, 0.975]).tolist()
                for i, o in enumerate(("AB", "BA"))
            },
            reverse_allocation=d.get("reverse_allocation", False),
            shared_workspace=d.get("shared_workspace", False),
            rounds=len(data),
        )
    result = dict(
        records=out,
        draws=args.draws,
        method="Resample complete eight-record paired rounds, retaining both frontends/orders/parities; percentile95%,seed20261118. Conditional on these graph/workspace allocations, not hardware-session population uncertainty. Identical-code controls are retained to expose systematic effects that these CIs do not capture.",
    )
    (root / (args.label + "-stage-pairs-summary.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("PASS", len(out), "paired stage records")


if __name__ == "__main__":
    main()

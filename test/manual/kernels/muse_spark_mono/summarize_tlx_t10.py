"""CPU summary of T10 cold headline pairs with paired-round bootstrap CIs."""

import argparse
import json
from pathlib import Path

import numpy as np


def pair_rounds(raw, reference):
    assert len(raw) % 8 == 0
    rounds = []
    for start in range(0, len(raw), 8):
        block = raw[start : start + 8]
        cells = {(r["label"], r["order"], r["parity"]): r["us"] for r in block}
        assert len(cells) == 8
        rounds.append(
            [
                [
                    [cells[(label, order, parity)] for parity in (0, 1)]
                    for order in ("AB", "BA")
                ]
                for label in (reference, "tlx")
            ]
        )
    return np.asarray(rounds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="phaseT10")
    parser.add_argument("--reference", default="t9")
    parser.add_argument("--sessions", nargs=2, default=("A", "B"))
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    root = Path("/home/bangtliu/coord/logs")
    rng = np.random.default_rng(20261118)
    rows = []
    for context, left in ((8192, -1), (8192, 2047), (131072, -1), (131072, 2047)):
        row = dict(context=context, left=left, pairs={})
        for reference in ("regular", args.reference, "gluon"):
            records = [
                json.loads(
                    (
                        root / f"{args.label}-{s}-headline-{context}-{left}.json"
                    ).read_text()
                )["pairs"][reference]
                for s in args.sessions
            ]
            samples = []
            for record in records:
                data = pair_rounds(record["raw"], reference)
                indices = rng.integers(0, len(data), (args.draws, len(data)))
                # Resample whole eight-record rounds, retaining paired labels,
                # both AB/BA orders and both epoch parities in each draw.
                medians = np.median(data[indices], axis=(1, 4))
                samples.append(medians[:, 1, :] / medians[:, 0, :])
            ratio_samples = np.mean(np.stack(samples), axis=(0, 2))
            ratios = [r["ratios"][o] for r in records for o in ("AB", "BA")]
            row["pairs"][reference] = dict(
                mean_ratio=float(np.mean(ratios)),
                ratio_ci95=np.quantile(ratio_samples, [0.025, 0.975]).tolist(),
                observed_ratio_range=[min(ratios), max(ratios)],
                per_session_orders={
                    s: r["ratios"] for s, r in zip(args.sessions, records)
                },
                reference_us=float(
                    np.mean(
                        [
                            r["per_copy_us"][reference][o]
                            for r in records
                            for o in ("AB", "BA")
                        ]
                    )
                ),
                tlx_us=float(
                    np.mean(
                        [
                            r["per_copy_us"]["tlx"][o]
                            for r in records
                            for o in ("AB", "BA")
                        ]
                    )
                ),
            )
        rows.append(row)
    result = dict(
        sessions=args.sessions,
        rows=rows,
        bootstrap_draws=args.draws,
        ci_method="Paired eight-record round bootstrap within each session; mean of the two sessions' AB/BA median ratios; percentile95%; seed20261118. CI reflects within-session precision, not a population of hardware sessions.",
    )
    (root / f"{args.label}-headline-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for row in rows:
        print(
            row["context"],
            row["left"],
            {r: d["mean_ratio"] for r, d in row["pairs"].items()},
        )


if __name__ == "__main__":
    main()

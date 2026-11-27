"""Balance graph order and collective epoch parity with AA/BB blocks."""

import argparse
import json
from pathlib import Path

import torch
from bench_chained47 import setup
from layer_fixture import require_gpu0
from paired45 import distribution

LOG = Path("/home/bangtliu/coord/logs")


def capture(prepare, call):
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare()
        start.record()
        call()
        end.record()
    return graph, start, end


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    a = p.parse_args()
    cases, native, ws, errors, routes = setup(a.context, a.left)
    epochs = {"regular": native.epochs[0], "chained": ws.s3.state.epochs}
    runs = {}
    for label, (prepare, call, check) in cases.items():
        prepare()
        call()
        torch.cuda.synchronize()
        check()
        runs[label] = capture(prepare, call)
    rows = []
    dummies = {n: 0 for n in cases}
    labels = tuple(cases)
    for rep in range(208):
        order = "AB" if rep % 2 == 0 else "BA"
        first_parity = (rep // 2) % 2
        # Balanced alignment replays are outside the measured samples. Never
        # rewrite an epoch independently of its arrival counters/payload flags.
        for label in labels if rep % 4 < 2 else labels[::-1]:
            if (int(epochs[label].flatten()[0]) + 1) % 2 != first_parity:
                runs[label][0].replay()
                torch.cuda.synchronize()
                dummies[label] += 1
        for label in labels if order == "AB" else labels[::-1]:
            for position in range(2):
                graph, start, end = runs[label]
                graph.replay()
                torch.cuda.synchronize()
                parity = int(epochs[label].flatten()[0]) % 2
                assert parity == (first_parity + position) % 2
                if rep >= 8:
                    rows.append(
                        dict(
                            label=label,
                            order=order,
                            parity=parity,
                            position=position,
                            us=start.elapsed_time(end) * 1000,
                        )
                    )
    result = {}
    for label in labels:
        cases[label][2]()
        result[label] = {
            f"{order}-parity{parity}": distribution(
                [
                    r["us"]
                    for r in rows
                    if r["label"] == label
                    and r["order"] == order
                    and r["parity"] == parity
                ]
            )
            for order in ("AB", "BA")
            for parity in (0, 1)
        }
        assert all(v["n"] == 100 for v in result[label].values())
    output = dict(
        context=a.context,
        left=a.left,
        table=result,
        raw=rows,
        alignment_replays=dummies,
        regular_errors=errors,
        routing=routes,
        protocol="AA/BB blocks; both first parities balanced; epoch readback outside events; no epoch/counter edits",
    )
    (LOG / f"phase48-parity-{a.context}-{a.left}.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

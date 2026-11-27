"""Interleaved graph measurement protocol: same process, both orders, raw data."""

import torch


def distribution(samples):
    s = sorted(samples)
    n = len(s)
    return dict(
        n=n,
        median_us=(s[(n - 1) // 2] + s[n // 2]) / 2,
        p99_us=s[max(0, (99 * n + 99) // 100 - 1)],
    )


def measure(cases, observer=None):
    runs = {}
    for label, (prepare, call, check) in cases.items():
        prepare()
        call()
        torch.cuda.synchronize()
        check()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepare()
            start.record()
            call()
            end.record()
        runs[label] = (graph, start, end, check, {"AB": [], "BA": []})
    labels = tuple(cases)
    assert len(labels) == 2
    for rep in range(210):
        order = "AB" if rep % 2 == 0 else "BA"
        for label in labels if order == "AB" else labels[::-1]:
            graph, start, end, check, samples = runs[label]
            graph.replay()
            torch.cuda.synchronize()
            if rep >= 10:
                samples[order].append(start.elapsed_time(end) * 1000)
                if observer is not None:
                    observer(label, order)
    result = {}
    for label, (graph, start, end, check, samples) in runs.items():
        check()
        assert all(len(v) == 100 for v in samples.values())
        result[label] = dict(
            **distribution(samples["AB"] + samples["BA"]),
            by_order={k: distribution(v) for k, v in samples.items()},
            samples_by_order_us=samples,
        )
    return result, {label: r[0] for label, r in runs.items()}

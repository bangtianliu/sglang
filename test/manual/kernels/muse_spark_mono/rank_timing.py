"""Measure a rank graph with stand-in preparation excluded from GPU event timing."""

import numpy as np
import torch


def rank_times(prepare, launch, *, samples=100):
    for _ in range(3):
        prepare()
        launch()
    torch.cuda.synchronize()
    setup = torch.cuda.CUDAGraph()
    with torch.cuda.graph(setup):
        prepare()
    stage = torch.cuda.CUDAGraph()
    with torch.cuda.graph(stage):
        launch()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for first, last in zip(starts, stops):
        setup.replay()
        first.record()
        stage.replay()
        last.record()
    torch.cuda.synchronize()
    times = [a.elapsed_time(b) * 1000 for a, b in zip(starts, stops)]
    return dict(
        median_us=float(np.median(times)),
        p99_us=float(np.quantile(times, 0.99)),
        standin_setup_excluded=True,
    )

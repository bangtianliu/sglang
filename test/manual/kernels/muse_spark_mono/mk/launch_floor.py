"""Launch floor on gfx1250 (GPU0): µs per launch of a kernel that only reads one word and exits,
by grid size and waves per CTA, in a CUDA graph of 200 launches (MK.79)."""
import json
import os
import time

import torch
import triton
import triton.language as tl


@triton.jit
def noop(Flag, Out):
    if tl.load(Flag) != 0:
        tl.store(Out + tl.program_id(0), 1)


def main():
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    flag = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = torch.zeros(4096, device="cuda", dtype=torch.int32)
    rows = []
    for warps in (1, 4):
        for grid in (1, 8, 64, 192, 288, 576, 1728):
            noop[(grid,)](flag, out, num_warps=warps)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(200):
                    noop[(grid,)](flag, out, num_warps=warps)
            best = 1e9
            for _ in range(10):
                torch.cuda.synchronize()
                t = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                best = min(best, (time.perf_counter() - t) / 200 * 1e6)
            rows.append(dict(warps=warps, grid=grid, us=round(best, 2)))
            del g
    print(json.dumps(rows))


if __name__ == "__main__":
    main()

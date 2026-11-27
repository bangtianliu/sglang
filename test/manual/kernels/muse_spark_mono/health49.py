"""Authorized fresh-process GPU0 recovery check; no peer memory or model imports."""

import json

import torch
from layer_fixture import require_gpu0

require_gpu0()
x = torch.arange(65536, device="cuda", dtype=torch.int64)
y = x * 3 + 7
torch.cuda.synchronize()
assert torch.equal(y.cpu(), torch.arange(65536, dtype=torch.int64) * 3 + 7)
print(
    json.dumps(
        {
            "gpu": 0,
            "elements": 65536,
            "integer_vector_exact": True,
            "device": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
        }
    ),
    flush=True,
)

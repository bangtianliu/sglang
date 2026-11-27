"""chain10 with the exact bf16 expert path (MX13 / SPLIT13 / FOLD13 off), for timing against the default."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_experts  # noqa: E402

tlx_experts.MX13 = tlx_experts.SPLIT13 = tlx_experts.FOLD13 = False
import chain10  # noqa: E402

chain10.main()

"""Snellius-only dataset override; install beside dataset_default.py.

The batch job sets a resolved, job-specific node-local path. Other dataset
paths keep their repository defaults. There is no persistent TMPDIR value.
"""

import os
from pathlib import Path

_imagenet21k_root = os.environ.get("LLR2_IMAGENET21K_ROOT")
if _imagenet21k_root:
    imagenet21k_p_path = Path(_imagenet21k_root).expanduser()

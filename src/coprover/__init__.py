"""
Significant files

lemmaret.bin.ir_exps: Canonical experiments for IR based evaluation

"""

import os
from pathlib import Path

PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RSC_ROOT = Path(PROJ_ROOT, "resources")
DATA_ROOT = Path(PROJ_ROOT, "data")

# Convenience paths
PVSDATA_ROOT = Path(DATA_ROOT, "pvs")
PVSLIB_ROOT = Path(PVSDATA_ROOT, "pvslib")
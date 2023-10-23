import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from coprover import PROJ_ROOT, RSC_ROOT

from coprover.feats.featurize_cmdpred import *

DATA_ROOT = Path(PROJ_ROOT, "data", "pvs", "prelude")
json_files = list(DATA_ROOT.rglob("*.json"))

import json
json_fpath = "/Users/yeh/proj/CoProver/data/pvs/prelude/EquivalenceClosure-proofs/EquivClosIdempotent.json"
with open(json_fpath, 'r') as f:
    json_root = json.load(f)

step = json_root['proof'][2]
tagform = convert_dict2tags(step)
state_str = format_state(tagform)

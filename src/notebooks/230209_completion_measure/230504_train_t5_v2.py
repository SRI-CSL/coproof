"""
Trains T5 against the v2 of the data, which prefixes all start tokens
"""

import pandas as pd
from pathlib import Path

from exp_setup import *
from coprover.training.simplet5 import SimpleT5
from sklearn.model_selection import train_test_split

DEBUG=False
print(f"Debug={DEBUG}")

if DEBUG:
    MAX_EPOCHS=1
else:
    MAX_EPOCHS=10

output_root = Path("outputs")
output_root.mkdir(exist_ok=True, parents=True)

MODELS_DIR = Path("outputs", "t5", "v2")

if DEBUG:
    MODELS_DIR = Path("outputs", "t5", "v2_debug")

# Model listing, tuple (strip_cmdhistory, cmds_only, path)
MODELS = {
    "t5_full_v2": (False, False, Path(MODELS_DIR, "laststep_pred_v2")),
    "t5_cmdsonly_v2": (False, True, Path(MODELS_DIR, "laststep_pred_cmdsonly_v2")),
    "t5_nocmds_v2": (True, False, Path(MODELS_DIR, "laststep_pred_nocmds_v2"))
}


def train_model(exp_name, strip_cmdhistory, cmds_only, model_fpath):
    print(f"Running experiment: {exp_name}, model_fpath={model_fpath}, strip_cmdhistory={strip_cmdhistory}, cmds_only={cmds_only}")
    train_df, test_df = setup_laststep_pred_data(csv_file="laststep_pred.v2.csv.gz",
        strip_cmdhistory=strip_cmdhistory, cmds_only=cmds_only)
    if DEBUG:
        train_df = train_df[0:100]
    model = SimpleT5(source_max_token_len=SRC_MAX_TOKLEN, target_max_token_len=TGT_MAX_TOKLEN)
    model.from_pretrained("t5", "t5-base")
    model.train(train_df=train_df, eval_df=test_df,
                max_epochs=MAX_EPOCHS, batch_size=6, 
                outputdir=model_fpath,
                save_only_last_epoch=False,
                num_gpus=[1,2,3,4], accelerator="ddp")

for exp_name, (strip_cmdhistory, cmds_only, model_fpath) in MODELS.items():
    train_model(exp_name, strip_cmdhistory, cmds_only, model_fpath)

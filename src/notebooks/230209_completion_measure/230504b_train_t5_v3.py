"""
Trains T5 against the v3 of the data, which does not include the start sequent
"""

import pandas as pd
from pathlib import Path

from exp_setup import *
from coprover.training.simplet5 import SimpleT5
from sklearn.model_selection import train_test_split

DEBUG=True
if DEBUG:
    GPUS = [0, 1]
    BATCH_SIZE = 1
else:
    BATCH_SIZE = 8
    GPUS = [1,2,3,4]
    
print(f"Debug={DEBUG}")

if DEBUG:
    MAX_EPOCHS=1
else:
    MAX_EPOCHS=10

output_root = Path("outputs")
output_root.mkdir(exist_ok=True, parents=True)

MODELS_DIR = Path("outputs", "t5", "v3")

if DEBUG:
    MODELS_DIR = Path("outputs", "t5", "v3_debug")

# Model listing, tuple (strip_cmdhistory, cmds_only, path)
MODELS = {
    "t5_full_v3": (False, False, Path(MODELS_DIR, "laststep_pred_v3")),
    "t5_cmdsonly_v3": (False, True, Path(MODELS_DIR, "laststep_pred_cmdsonly_v3")),
    "t5_nocmds_v3": (True, False, Path(MODELS_DIR, "laststep_pred_nocmds_v3"))
}


def train_model(exp_name, strip_cmdhistory, cmds_only, model_fpath):
    print(f"Running experiment: {exp_name}, model_fpath={model_fpath}, strip_cmdhistory={strip_cmdhistory}, cmds_only={cmds_only}")
    train_df, test_df = setup_laststep_pred_data(csv_file="laststep_pred.v3.csv.gz",
        strip_cmdhistory=strip_cmdhistory, cmds_only=cmds_only)
    if DEBUG:
        train_df = train_df[0:100]
    model = SimpleT5(source_max_token_len=SRC_MAX_TOKLEN, target_max_token_len=TGT_MAX_TOKLEN)
    model.from_pretrained("t5", "t5-base")
    model.train(train_df=train_df, eval_df=test_df,
                max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, 
                outputdir=model_fpath,
                save_only_last_epoch=False,
                num_gpus=GPUS, accelerator="ddp")

for exp_name, (strip_cmdhistory, cmds_only, model_fpath) in MODELS.items():
    train_model(exp_name, strip_cmdhistory, cmds_only, model_fpath)

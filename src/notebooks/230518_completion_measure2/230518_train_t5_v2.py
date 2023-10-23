import pandas as pd
from pathlib import Path

from exp_setup import *
from coprover.training.simplet5 import SimpleT5
from sklearn.model_selection import train_test_split

GPUS = [1,5,6]

output_root = Path("outputs")
output_root.mkdir(exist_ok=True, parents=True)

inst_df = pd.read_csv("laststep_pred.v2.csv.gz")
train_df, test_df = train_test_split(inst_df, random_state=501, test_size=0.1)

model = SimpleT5(source_max_token_len=SRC_MAX_TOKLEN, target_max_token_len=TGT_MAX_TOKLEN)
model.from_pretrained("t5", "t5-base")

model.train(train_df=train_df, eval_df=test_df,
            max_epochs=10, batch_size=8, 
            outputdir="outputs/laststep_pred_v2",
            save_only_last_epoch=False,
            num_gpus=GPUS, accelerator="ddp")

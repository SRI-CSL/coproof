"""
Same as 230209_train_t5_v1, but without command history
"""

import pandas as pd
from pathlib import Path

from exp_setup import *
from coprover.training.simplet5 import SimpleT5
from sklearn.model_selection import train_test_split

output_root = Path("outputs")
output_root.mkdir(exist_ok=True, parents=True)

train_df, test_df = setup_laststep_pred_data(strip_cmdhistory=False, cmds_only=True)

print("Training sample:")
for r in train_df.source_text.values[0:3]:
    print(r[0:100])

model = SimpleT5(source_max_token_len=SRC_MAX_TOKLEN, target_max_token_len=TGT_MAX_TOKLEN)
model.from_pretrained("t5", "t5-base")

model.train(train_df=train_df, eval_df=test_df,
            max_epochs=10, batch_size=1, 
            outputdir="outputs/laststep_pred_cmdsonly_v1",
            save_only_last_epoch=False,
            num_gpus=[0,1], accelerator="ddp")

import pandas as pd
from pathlib import Path

from exp_setup import *
from coprover.training.simplet5 import SimpleT5
from sklearn.model_selection import train_test_split

DEBUG=False
USE_CMDHIST=True

if DEBUG:
    GPUS = [0, 1]
    BATCH_SIZE = 1
else:
    GPUS = [0, 1]
    BATCH_SIZE = 4

if USE_CMDHIST:
    OUTPUT_DIR = "outputs/compare_pred_v1"
else:
    OUTPUT_DIR = "outputs/compare_pred_v1.nocmdhist"
    
output_root = Path("outputs")
output_root.mkdir(exist_ok=True, parents=True)

train_df, test_df = setup_data(debug=DEBUG, strip_cmdhistory=not(USE_CMDHIST))

model = SimpleT5(source_max_token_len=SRC_MAX_TOKLEN, target_max_token_len=TGT_MAX_TOKLEN)
model.from_pretrained("t5", "t5-base")

model.train(train_df=train_df, eval_df=test_df,
            max_epochs=10, batch_size=BATCH_SIZE, 
            outputdir=OUTPUT_DIR,
            save_only_last_epoch=False,
            num_gpus=GPUS, accelerator="ddp")

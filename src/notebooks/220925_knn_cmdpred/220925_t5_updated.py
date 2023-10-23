# Updated T5 model using PVSlib subsample split

import pandas as pd
from coprover.training.simplet5 import SimpleT5
from sklearn.model_selection import train_test_split
from pathlib import Path

from data_setup import *


train_df, test_df = get_full_dfs()

MODEL_OUTDIR = Path("outputs/cmd_pred1_N3")
MODEL_OUTDIR.mkdir(exist_ok=True, parents=True)

# Get max on sentence lengths
max_src_tok_len = max([len(x.split()) for x in full_df['source_text']]) + 10

print("Max source toklength={}".format(max_src_tok_len))

# Prediction task, use minimal
model = SimpleT5(source_max_token_len=max_src_tok_len,
                 target_max_token_len=10)
model.from_pretrained("t5", "t5-base")

model.train(train_df=train_df,
                eval_df=test_df,
                max_epochs=10,
                batch_size=2,
                dataloader_num_workers=4,
                outputdir=MODEL_OUTDIR,
                save_only_last_epoch=True,
                num_gpus=1)

print(model.predict(
    CMD1_PREFIX + "<ANT> NOOP NOOP NOOP <pad> <CONS> s-formula forall ['variable'] ['variable'] apply constant type-actual apply constant type-actual type-actual apply constant ['variable'] ['variable'] apply constant apply constant type-actual type-actual ['variable'] apply constant type-actual type-actual ['variable'] <HID>"))

from tqdm import tqdm


def score_df(df):
    num_correct = 0
    total = 0
    guesses = []
    golds = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        guess = model.predict(row['source_text'])[0]
        gold = row['target_text']
        guesses.append(guess)
        golds.append(gold)
        total += 1
        if guess == gold:
            num_correct += 1
    return num_correct / total


print("Train acc={:.3f}".format(score_df(train_df)))
print("Test acc={:.3f}".format(score_df(test_df)))

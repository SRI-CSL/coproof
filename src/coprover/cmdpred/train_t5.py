"""
Trains the T5 backed predictor, using the following from src/notebooks/220325_t5:

- 220325_data_setup.py
"""

"""Applies simplet5 training, but uses the N3 command history as a feature

Be sure to run 220325_data_setup.py first to featurize the proof data
into the correct form.

"""
import pandas as pd
from coprover.training.simplet5 import SimpleT5
from coprover.cmdpred.cmdpred_data_setup import get_full_dfs, get_sequentonly_dfs, get_cmdhistonly_dfs, max_src_tok_len, SRC_TXT, TGT_TXT, CMD_HIST
from sklearn.model_selection import train_test_split
from pathlib import Path

USE_MLM = False

FULL = "full"
SEQONLY = "sequentonly"
CMDHISTONLY = "cmdhistonly"

EXPTYPE = CMDHISTONLY
print("Experiment type={}".format(EXPTYPE))
outputdir = Path("outputs")

outputdir.mkdir(exist_ok=True, parents=True)

if EXPTYPE == FULL:
    train_df, test_df = get_full_dfs()
elif EXPTYPE == SEQONLY:
    train_df, test_df = get_sequentonly_dfs()
elif EXPTYPE == CMDHISTONLY:
    train_df, test_df = get_cmdhistonly_dfs()

print("Size train={}, test={}".format(len(train_df), len(test_df)))
print("Max source toklength={}".format(max_src_tok_len))

# Prediction task, use minimal
model = SimpleT5(source_max_token_len=max_src_tok_len,
                 target_max_token_len=10)
model.from_pretrained("t5", "t5-base")

CACHED_FPATH = Path("models", "cmdprec_N3", "curr_best")

model.train(train_df=train_df,
            eval_df=test_df,
            max_epochs=10,
            batch_size=8,
            dataloader_num_workers=4,
            outputdir="outputs/pvslib_cmdpred_t5_{}".format(EXPTYPE),
            save_only_last_epoch=True,
            num_gpus=2
)

print(model.predict(
    CMD1_PREFIX + "<ANT> NOOP NOOP NOOP <pad> <CONS> s-formula forall ['variable'] ['variable'] apply constant type-actual apply constant type-actual type-actual apply constant ['variable'] ['variable'] apply constant apply constant type-actual type-actual ['variable'] apply constant type-actual type-actual ['variable'] <HID>"))

from tqdm import tqdm


def score_df(df):
    num_correct = 0
    total = 0
    guesses = []
    golds = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        guess = model.predict(row[SRC_TXT])[0]
        gold = row[TGT_TXT]
        guesses.append(guess)
        golds.append(gold)
        total += 1
        if guess == gold:
            num_correct += 1
    return num_correct / total


print("Train acc={:.3f}".format(score_df(train_df)))
print("Test acc={:.3f}".format(score_df(test_df)))

# TODO Confusion

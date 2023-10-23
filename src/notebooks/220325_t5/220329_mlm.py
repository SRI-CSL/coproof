"""
Explore using the prelude data to conduct a MLM task
If successful, this wil segue into the multi-task learning framework

This requires the data to be setup by 220325_data_setup.py first.
"""
import torch
import numpy as np
import pandas as pd
from simplet5 import SimpleT5
from sklearn.model_selection import train_test_split
import multiprocessing

from coprover.utils import count_freqs

# DF Labels
SOURCE_TEXT = "source_text"
TARGET_TEXT = "target_text"

# Command prefixes
MLM1_PREFIX="MLM1: "

model = SimpleT5()
model.from_pretrained("t5", "t5-base")

full_df = pd.read_csv("tags_input_pred_pairs.tsv",
                      sep="\t",
                      header=None,
                      names=[SOURCE_TEXT,
                             TARGET_TEXT])



# Now change the target text to match the source text
full_df[TARGET_TEXT] = full_df[SOURCE_TEXT]

# Now attach the command prefix
full_df[SOURCE_TEXT] = MLM1_PREFIX + full_df[SOURCE_TEXT]

# Split apart sentences by train and test
# TODO: Consider hardwiring library to accept inputs via function, instead of
# via a dataframe
train_df, test_df = train_test_split(full_df, test_size=0.1,
                                     random_state=1337,
                                     shuffle=True)

# Generate multiple replicates of the training rows, so we get exposure to
# all of the tokens
train_df = pd.concat([train_df] * 10, ignore_index=True)

# Go through each source text and ablate a token
rng = np.random.default_rng(343)

ANT_TOK = "<ANT>"
CONS_TOK = "<CONS>"
HIDDEN_TOK = "<HID>"
MASK_TOK = "<MASK>"

RESV_WORDS = set([ANT_TOK, CONS_TOK, HIDDEN_TOK, "MLM:"])

def ablate_toks(seq, N=1):
    toks = seq.split()
    indices = np.arange(0, len(toks))
    rng.shuffle(indices)
    masked_idxes = []
    for idx in indices:
        tok = toks[idx]
        if tok not in RESV_WORDS and len(masked_idxes) < N:
            masked_idxes.append(idx)
    ret = []
    for idx, tok in enumerate(toks):
        if idx in masked_idxes:
            ret.append(MASK_TOK)
        else:
            ret.append(tok)
    return " ".join(ret)


import math
def ablate_toks_frac(seq, frac=0.15):
    N = math.ceil(len(seq.split()) * frac)
    return ablate_toks(seq, N=N)
    

# Go through each of the train rows and ablate
# TODO: Create several replicates of the original rows and
# ablate from those as well.
for idx, row in train_df.iterrows():
    row[SOURCE_TEXT] = ablate_toks(row[SOURCE_TEXT])



# Get max on sentence lengths
max_src_tok_len = max([len(x.split()) for x in full_df[SOURCE_TEXT]])+ 10

print("Max source toklength={}".format(max_src_tok_len))

if False:
    model.train(train_df=train_df,
                eval_df=test_df,
                max_epochs=10,
                batch_size=4,
                dataloader_num_workers=4,
                source_max_token_len=max_src_tok_len,
                target_max_token_len=max_src_tok_len,
                save_only_last_epoch=True,
                use_gpu=True)

if True:
    MODEL_FPATH = "models/mlm_N1/simplet5-epoch-9-train-loss-0.0005-val-loss-0.0"
    model.load_model("t5", MODEL_FPATH, use_gpu=True)

def test(idx):
    tgt_txt = test_df.iloc[idx][TARGET_TEXT]
    src_txt = ablate_toks(test_df.iloc[idx][SOURCE_TEXT], N=2)    
    #src_txt = ablate_toks_frac(test_df.iloc[idx][SOURCE_TEXT])
    gold = tgt_txt.split()
    guess = model.predict(src_txt, max_length=700)[0].split()
    num_correct=0
    total = max(len(gold), len(guess))
    num_correct = 0
    for gu, go in zip(guess, gold):
        if gu.startswith("<"):
            gu = gu[1:] # Remove first <, as tokenizer removes it
        if gu == go:
            num_correct += 1
    return num_correct / total

def scan(X):
    mu, std, xmin, xmax = np.mean(X), np.std(X), np.min(X), np.max(X)
    return "mu/std={:.3f}/{:.3f}, min/max={:.3f}/{:.3f}".format(mu, std, xmin, xmax)


from tqdm import tqdm
accs = []
tqdm_iter = tqdm(range(len(test_df)))
for idx in tqdm_iter:
    last_acc = test(idx)
    accs.append(last_acc)
    curr_mean_acc = np.mean(accs)
    curr_std_acc = np.std(accs)    
    tqdm_iter.set_description("last acc={:.3f}, mean/std={:.3f}/{:.3f}".format(last_acc, curr_mean_acc, curr_std_acc))
print(scan(accs))


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

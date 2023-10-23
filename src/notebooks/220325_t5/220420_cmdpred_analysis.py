#!/usr/bin/env python
# coding: utf-8

# # Command Prediction Analysis
# 
# Mirrors mostly results in 220419_cmdpred_analysis.py, but also does some more graphing.

# In[1]:


"""
Loads in the T5 simple command prediction and performs some simple analyses

Because we're using the pre-trained tokenizer, we may get fragments of commands from
the output.  Until we can train up our own vocabulary, we'll use a prefix based
identifier organized by frequency
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
from coprover.training.simplet5 import SimpleT5
from sklearn.model_selection import train_test_split
from pathlib import Path
from matplotlib import pyplot as plt

from coprover.feats.commands import normalize
from coprover.utils import count_freqs

USE_GPU = True  # TODO: Set based on number of GPUs

USE_CONDMLM_MODEL = False  # If True, uses the MLM conditioned model, otherwise from scratch

# Command prefixes are always expected
CMD1_PREFIX = "command1: "
SRC_TXT = 'source_text'
TGT_TXT = 'target_text'

full_df = pd.read_csv("tags_input_pred_pairs.tsv",
                      sep="\t",
                      header=None,
                      names=['source_text',
                             'target_text'])

full_df['source_text'] = CMD1_PREFIX + full_df['source_text']


unique_commands = full_df['target_text'].unique()

train_df, test_df = train_test_split(full_df, test_size=0.1,
                                     random_state=1337,
                                     shuffle=True)

# Until we get full discretre vocab in, we'll use a frequency ordered
# prefix table
cmd_freqs = count_freqs(train_df['target_text'])
test_cmd_freqs = count_freqs(test_df['target_text'])

# Get max on sentence lengths
max_src_tok_len = max([len(x.split()) for x in full_df['source_text']]) + 10

print("Max source toklength={}".format(max_src_tok_len))

# Prediction task, use minimal
model = SimpleT5(source_max_token_len=max_src_tok_len,
                 target_max_token_len=10)

if USE_CONDMLM_MODEL:
    CACHED_FPATH = Path("models", "cmdprec_mlmn1", "curr_best")
    expname = "w_MLM"
else:
    CACHED_FPATH = Path("models", "cmd_pred1_noMLM2", "curr_best")
    expname="no_MLM"
print("Model used = {}".format(CACHED_FPATH))
model.load_model(model_type="t5", model_dir=CACHED_FPATH, use_gpu=USE_GPU)

test_query = CMD1_PREFIX + "<ANT> <CONS> s-formula forall ['variable'] ['variable'] apply constant type-actual apply constant type-actual type-actual apply constant ['variable'] ['variable'] apply constant apply constant type-actual type-actual ['variable'] apply constant type-actual type-actual ['variable'] <HID>"
N=10
pred_res = model.predict(test_query, num_return_sequences=N, num_beams=2*N)
for idx, cmd in enumerate(pred_res):
    print(idx, normalize(cmd))


def acc_at_N(df, N=5):
    """ Gets accuracy @ N"""
    tp = 0
    tqdm_iter = tqdm(df.iterrows(), total=len(df))
    num_seen = 0
    for idx, row in tqdm_iter:
        gold = row[TGT_TXT]
        guesses = [normalize(pref) for pref in model.predict(row[SRC_TXT], num_return_sequences=N, num_beams=2 * N)]
        if gold in guesses:
            tp += 1
        num_seen += 1
        tqdm_iter.set_description("{}/{}, {:.3f}".format(tp, num_seen, tp/num_seen))
    return tp / len(df)

results_at_N = []
for N in range(1, 11):
    acc = acc_at_N(test_df, N=N)
    print("{}:\t{:.3f}".format(N, acc))
    results_at_N.append((N, acc))

plt.clf()
fig, ax = plt.subplots(1, figsize=(8, 6))
at_N = np.array([x[0] for x in results_at_N])
accs = np.array([x[1] for x in results_at_N])
ax.plot(at_N, accs, label="Acc@N")
plt.xticks(ticks=at_N)
plt.legend()
plt.title("Test Accuracy @ N")
plt.savefig("test_acc_at_N.{}.png".format(expname))

plt.clf()
train_results_at_N = []
for N in range(1, 11):
    acc = acc_at_N(train_df, N=N)
    print("{}:\t{:.3f}".format(N, acc))
    train_results_at_N.append((N, acc))

fig, ax = plt.subplots(1, figsize=(8, 6))
at_N = np.array([x[0] for x in train_results_at_N])
accs = np.array([x[1] for x in train_results_at_N])
ax.plot(at_N, accs, label="Acc@N")
plt.xticks(ticks=at_N)
plt.legend()
plt.title("Train Accuracy @ N")
plt.savefig("train_acc_at_N.{}.png".format(expname))


# Save and plot both
plt.clf()
fig, ax = plt.subplots(1, figsize=(8, 6))
at_N = np.array([x[0] for x in train_results_at_N])
train_accs = np.array([x[1] for x in train_results_at_N])
test_accs = np.array([x[1] for x in results_at_N])
combined_df = pd.DataFrame({"N": at_N, "train_acc": train_accs, "test_accs": test_accs})
combined_df.to_csv("acc_at_N.{}.csv".format(expname), header=True, index=False)


plt.clf()
ax.plot(at_N, train_accs, color="blue", label="train")
ax.plot(at_N, test_accs, color="red", label="test")
ax.set_xlabel("N")
ax.set_ylabel("Accuracy")
ax.set_ylim([0, 1])
plt.xticks(ticks=at_N)
plt.legend()
plt.title("CmdPred Accuracies @ N")
plt.savefig("acc_at_N.{}.png".format(expname))


# In[19]:


row = test_df.iloc[15]
num_return = 1
for nbeams in [1,5,10,20,50,100]:
    if num_return <= nbeams:
        preds = model.predict(row[SRC_TXT], num_return_sequences=num_return, num_beams=nbeams)
        print(nbeams, preds)


# In[20]:


row = test_df.iloc[15]
num_return = 2
for nbeams in [1,5,10,20,50,100]:
    if num_return <= nbeams:
        preds = model.predict(row[SRC_TXT], num_return_sequences=num_return, num_beams=nbeams)
        print(nbeams, preds)


# In[21]:


row = test_df.iloc[15]
num_return = 3
for nbeams in [1,5,10,20,50,100]:
    if num_return <= nbeams:
        preds = model.predict(row[SRC_TXT], num_return_sequences=num_return, num_beams=nbeams)
        print(nbeams, preds)


# In[22]:


row = test_df.iloc[15]
num_return = 5
for nbeams in [1,5,10,20,50,100]:
    if num_return <= nbeams:
        preds = model.predict(row[SRC_TXT], num_return_sequences=num_return, num_beams=nbeams)
        print(nbeams, preds)


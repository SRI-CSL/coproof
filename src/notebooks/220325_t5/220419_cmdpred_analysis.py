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

from coprover.utils import count_freqs

USE_GPU = True  # TODO: Set based on number of GPUs

USE_CONDMLM_MODEL = True  # If True, uses the MLM conditioned model, otherwise from scratch

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

# Unify commands
# 'instantiate' -> 'inst' (more frequent)

def normalize_cmd(cmd):
    if cmd == "instantiate":
        return "inst"
    return cmd

unique_commands = full_df['target_text'].unique()

train_df, test_df = train_test_split(full_df, test_size=0.1,
                                     random_state=1337,
                                     shuffle=True)

# Until we get full discretre vocab in, we'll use a frequency ordered
# prefix table
cmd_freqs = count_freqs(train_df['target_text'])
test_cmd_freqs = count_freqs(test_df['target_text'])


def get_cmd(prefix):
    for cmd, _ in cmd_freqs:
        if cmd.startswith(prefix):
            return normalize_cmd(cmd)
    raise Exception("Unknown command prefix={}".format(prefix))

# Get max on sentence lengths
max_src_tok_len = max([len(x.split()) for x in full_df['source_text']]) + 10

print("Max source toklength={}".format(max_src_tok_len))

# Prediction task, use minimal
model = SimpleT5(source_max_token_len=max_src_tok_len,
                 target_max_token_len=10)

if USE_CONDMLM_MODEL:
    CACHED_FPATH = Path("models", "cmdprec_mlmn1", "curr_best")
else:
    CACHED_FPATH = Path("models", "cmdprec_mlmn1", "curr_best")
model.load_model(model_type="t5", model_dir=CACHED_FPATH, use_gpu=USE_GPU)

test_query = CMD1_PREFIX + "<ANT> <CONS> s-formula forall ['variable'] ['variable'] apply constant type-actual apply constant type-actual type-actual apply constant ['variable'] ['variable'] apply constant apply constant type-actual type-actual ['variable'] apply constant type-actual type-actual ['variable'] <HID>"
N=10
pred_res = model.predict(test_query, num_return_sequences=N, num_beams=2*N)
for idx, cmd in enumerate(pred_res):
    print(idx, get_cmd(cmd))

def acc_at_N(df, N=5):
    """ Gets accuracy @ N"""
    tp = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        gold = row[TGT_TXT]
        guesses = [get_cmd(pref) for pref in model.predict(row[SRC_TXT], num_return_sequences=N, num_beams=2*N)]
        if gold in guesses:
            tp += 1
    return tp / len(df)

results_at_N = []
for N in range(1, 11):
    acc = acc_at_N(test_df, N=N)
    print("{}:\t{:.3f}".format(N, acc))
    results_at_N.append((N, acc))


from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, figsize=(8, 6))
at_N = np.array([x[0] for x in results_at_N])
accs = np.array([x[1] for x in results_at_N])
ax.plot(at_N, accs, label="Acc@N")
plt.xticks(ticks=at_N)
plt.legend()
plt.title("Test Accuracy @ N")
plt.savefig("test_acc_at_N.png")


train_results_at_N = []
for N in range(1, 11):
    acc = acc_at_N(train_df, N=N)
    print("{}:\t{:.3f}".format(N, acc))
    train_results_at_N.append((N, acc))

from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, figsize=(8, 6))
at_N = np.array([x[0] for x in train_results_at_N])
accs = np.array([x[1] for x in train_results_at_N])
ax.plot(at_N, accs, label="Acc@N")
plt.xticks(ticks=at_N)
plt.legend()
plt.title("Train Accuracy @ N")
plt.savefig("train_acc_at_N.png")


# Save and plot both
from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, figsize=(8, 6))
at_N = np.array([x[0] for x in train_results_at_N])
train_accs = np.array([x[1] for x in train_results_at_N])
test_accs = np.array([x[1] for x in results_at_N])
combined_df = pd.DataFrame({"N": at_N, "train_acc": train_accs, "test_accs": test_accs})
combined_df.to_csv("acc_at_N.csv", header=True, index=False)


ax.plot(at_N, train_accs, color="blue", label="train")
ax.plot(at_N, test_accs, color="red", label="test")
ax.set_xlabel("N")
ax.set_ylabel("Accuracy")
plt.xticks(ticks=at_N)
plt.legend()
plt.title("CmdPred Accuracies @ N")
plt.savefig("acc_at_N.png")

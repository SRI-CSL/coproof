import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

from coprover import RSC_ROOT
from coprover.utils import count_freqs


# In[2]:


USE_MLM = False

# Command prefixes are always expected
CMD1_PREFIX = "command1: "

SRC_TXT = 'source_text'
TGT_TXT = 'target_text'
CMD_HIST = 'cmd_history'
BRANCH = 'branch'
DEPTH = 'depth'

# DATA_FPATH = Path(RSC_ROOT, "pvs_cmd_pred", "data", "cmdpred_N3.prelude.tsv.gz")
DATA_FPATH = Path(RSC_ROOT, "pvs_cmd_pred", "data", "cmdpred_N3.pvslib.tsv.gz")
#DATA_FPATH = Path(RSC_ROOT, "pvs_cmd_pred", "data", "cmdpred_N3.pvslib.pretty.tsv.gz")

full_df = pd.read_csv(DATA_FPATH,
                      sep="\t",
                      header=None,
                      names=[SRC_TXT,
                             TGT_TXT,
                             CMD_HIST,
                             BRANCH,
                             DEPTH])

# Subsample the DF to limit train/test times
full_df = full_df.sample(n=20000, random_state=42)

# Use full command history, with cmdhist as a single tok

def get_full_dfs():
    full_df[SRC_TXT] = CMD1_PREFIX + full_df[CMD_HIST].replace(",", "") + " <pad> " + full_df[SRC_TXT]
    train_df, test_df = train_test_split(full_df, test_size=0.1,
                                         random_state=1337,
                                         shuffle=True)
    return train_df, test_df


def get_sequentonly_dfs():
    full_df['source_text'] = CMD1_PREFIX +  " <pad> " + full_df[SRC_TXT]
    train_df, test_df = train_test_split(full_df, test_size=0.1,
                                         random_state=1337,
                                         shuffle=True)
    return train_df, test_df


def get_cmdhistonly_dfs():
    full_df['source_text'] = CMD1_PREFIX + full_df[CMD_HIST]
    train_df, test_df = train_test_split(full_df, test_size=0.1,
                                         random_state=1337,
                                         shuffle=True)
    return train_df, test_df



def normalize_cmd(cmd):
    if cmd == "instantiate":
        return "inst"
    return cmd

# Until we get full discretre vocab in, we'll use a frequency ordered
# prefix table
train_cmd_freqs = count_freqs(get_cmdhistonly_dfs()[0]['target_text'])

def get_cmd(prefix):
    """ Try and get the command based off of prefix.
    If at first try and does not grab, try with 
    just the 0:N-1 instead.  Keeps going until it automatically
    returns default of top guess. """
    for cmd, _ in train_cmd_freqs:
        if cmd.startswith(prefix):
            return normalize_cmd(cmd)
    return get_cmd(prefix[0:len(prefix) - 1])

    # Keep the exception as a sanity check
    raise Exception("Unknown command prefix={}".format(prefix))



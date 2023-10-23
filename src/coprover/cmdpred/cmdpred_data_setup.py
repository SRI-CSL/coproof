import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import collections
from sklearn.feature_extraction.text import TfidfVectorizer

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

orig_df = pd.read_csv(DATA_FPATH,
                      sep="\t",
                      header=None,
                      names=[SRC_TXT,
                             TGT_TXT,
                             CMD_HIST,
                             BRANCH,
                             DEPTH])

# Subsample the DF to limit train/test times
orig_subsample_df = orig_df.sample(n=20000, random_state=42)
max_src_tok_len = min(1000, max([len(x.split()) for x in orig_subsample_df['source_text']]) + 10)

print("Max source toklength={}".format(max_src_tok_len))

# Use full command history, with cmdhist as a single tok

def get_full_dfs(df=orig_subsample_df):
    tmp_df = df.copy()
    tmp_df['source_text'] = CMD1_PREFIX + df[CMD_HIST].replace(",", "") + " <pad> " + df[SRC_TXT]
    train_df, test_df = train_test_split(tmp_df, test_size=0.1,
                                         random_state=1337,
                                         shuffle=True)
    return train_df, test_df


def get_sequentonly_dfs(df=orig_subsample_df):
    tmp_df = df.copy()
    tmp_df['source_text'] = CMD1_PREFIX +  " <pad> " + df[SRC_TXT]
    train_df, test_df = train_test_split(tmp_df, test_size=0.1,
                                         random_state=1337,
                                         shuffle=True)
    return train_df, test_df


def get_cmdhistonly_dfs(df=orig_subsample_df):
    tmp_df = df.copy()
    tmp_df['source_text'] = CMD1_PREFIX + df[CMD_HIST]
    train_df, test_df = train_test_split(tmp_df, test_size=0.1,
                                         random_state=1337,
                                         shuffle=True)
    return train_df, test_df


FULL = "full"
SEQONLY = "sequentonly"
CMDHISTONLY = "cmdhistonly"

def get_splits(exp_type):
    """ Gets the experimental splits, given the type of experiment to run
    """
    if exp_type == FULL:
        train_df, test_df = get_full_dfs()
    elif exp_type == SEQONLY:
        train_df, test_df = get_sequentonly_dfs()
    elif exp_type == CMDHISTONLY:
        train_df, test_df = get_cmdhistonly_dfs()
    else:
        raise Exception("Unknown experiment type={}".format(exp_type))
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


def convert2XY(train_df, test_df, src_txt_vectorizer=None):
    """
    Formats the given train and test DataFrame using the given source text vectorizer
    and creates train/test XY matrices suitable for use with sklearn.
    """
    label_lookup = collections.OrderedDict()

    train_corpus = []
    train_targets = []
    for row in train_df.iterrows():
        src_txt = row[1][SRC_TXT]
        cmd = row[1][TGT_TXT]
        train_corpus.append(src_txt)
        train_targets.append(cmd)

    test_corpus = []
    test_targets = []
    for row in test_df.iterrows():
        src_txt = row[1][SRC_TXT]
        cmd = row[1][TGT_TXT]
        test_corpus.append(src_txt)
        test_targets.append(cmd)

    if src_txt_vectorizer is None:
        src_txt_vectorizer = TfidfVectorizer(stop_words=None)
    label_lookup = collections.OrderedDict()

    train_X = src_txt_vectorizer.fit_transform(train_corpus).toarray()
    test_X = src_txt_vectorizer.transform(test_corpus).toarray()

    # Combine train and test targets, so we get full coverage of labels
    for target in train_targets + test_targets:
        if target not in label_lookup:
            label_lookup[target] = len(label_lookup)

    def transform_labels(targets):
        Y = np.array([label_lookup[target] for target in targets])
        return Y

    train_Y = transform_labels(train_targets)
    test_Y = transform_labels(test_targets)

    # Now perform the sanity check
    for idx in range(len(train_targets)):
        label = train_targets[idx]
        label_id = train_Y[idx]
        assert(label_lookup[label] == label_id)
    return train_X, train_Y, test_X, test_Y, label_lookup

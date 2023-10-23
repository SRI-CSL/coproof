"""
Common experiment setup across notebooks.  May or may not make it to main codebase.
"""

import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

POS = "p"
NEG = "n"

SRC_MAX_TOKLEN = 1000
TGT_MAX_TOKLEN = 3


#
# Routines for setting up traditional machine learning experiments
#
def setup_data(csv_file="compare_pred.v1.csv.gz", strip_cmdhistory=False, cmds_only=False, debug=False):
    print(f"Setting up last step pred from {csv_file}")
    inst_df = pd.read_csv(csv_file)
    if debug:
        inst_df = inst_df[0:100]
    if strip_cmdhistory:
        # Remove command history from source_text
        updated_rows = [" ".join(row.split()[3:]) for row in inst_df.source_text.values]
        inst_df.source_text = updated_rows
    elif cmds_only:
        updated_rows = [" ".join(row.split()[0:3]) for row in inst_df.source_text.values]
        inst_df.source_text = updated_rows
    train_df, test_df = train_test_split(inst_df, random_state=501, test_size=0.1)
    print("Len train={}, test={}".format(len(train_df), len(test_df)))
    return train_df, test_df


def setup_cmd_dict_data():
    """
    Sets up the numeric X,Y arrays for command history standard classification methods experiment.

    TODO: adjust positives
    """
    train_df, test_df = setup_data()
    Y_train = np.zeros(len(train_df))
    Y_train[train_df.target_text == 'pos'] = 1
    Y_test = np.zeros(len(test_df))
    Y_test[test_df.target_text == 'pos'] = 1
    def featurize_cmds(df):
        datums = []
        for cmds in df.cmd_history.array:
            datum = {}
            for idx, cmd in enumerate(cmds.split()):
                datum["{}_{}".format(cmd, idx)] = 1
            datums.append(datum)
        return datums
    cmd_vectorizer = DictVectorizer(sparse=False)
    train_datums = featurize_cmds(train_df)
    test_datums = featurize_cmds(test_df)
    cmd_vectorizer.fit(train_datums)
    X_train = cmd_vectorizer.transform(train_datums)
    X_test = cmd_vectorizer.transform(test_datums)
    return X_train, Y_train, X_test, Y_test


def setup_state_dict_data(csv_file="laststep_pred.v2.csv.gz", use_tfidf=True):
    """
    Sets up the numeric X,Y arrays for command history standard classification methods experiment.
    """
    train_df, test_df = setup_data(csv_file=csv_file)
    Y_train = np.zeros(len(train_df))
    Y_train[train_df.target_text == 'pos'] = 1
    Y_test = np.zeros(len(test_df))
    Y_test[test_df.target_text == 'pos'] = 1
    def featurize_state(df):
        datums = []
        for raw_state in df.source_text.array:
            raw_state = " ".join(raw_state.split()[3:]) # Get everything but the commands
            datums.append(raw_state)
        return datums
    if use_tfidf:   
        tfidf_vectorizer = TfidfVectorizer()
    else:
        tfidf_vectorizer = CountVectorizer()
    train_datums = featurize_state(train_df)
    test_datums = featurize_state(test_df)
    tfidf_vectorizer.fit(train_datums)
    X_train = tfidf_vectorizer.transform(train_datums).todense()
    X_test = tfidf_vectorizer.transform(test_datums).todense()
    return X_train, Y_train, X_test, Y_test

#!/usr/bin/env python
# coding: utf-8

# %load_ext autoreload
# %autoreload 2


# This is an update using the corrected TheoryBank
# This is an extension to 220707_lemma_retrieval_exps, except this performs checks over feature deltas
#
#
# ORIGINAL COPY
#
# Similar to the first lemma retrieval experiment, but we now use a CountVectorizer and TfidfVectorizer to properly convert everything into vectors.
#
# We take in $N lemma requests from the command history.  A lemma request is a PVS *lemma* command where the first argument is the lemma name to import.  We take these from the command history for each of the proofs.
#
# In order to simplify the problem, we pool all of the possible lemmas from all theories and consider them all as candidates.  Future iterations of this experiment will consider restricting the set of considered lemmas to be just the ones available in the theory the proof was derived for in PVSLib.
#
# Another todo is to incorporate the PVS Prelude, as that is a mutually exclusive set of theories and proofs.
#
# To faciliate comparisons against supervised methods, we separate the lemma queries into train and test sets, randomly shuffled and partitioned at a 60/40 train/test split.  From an original set of 20221 viable lemma queries, we get a train/test split size of 12132/8089.
#
# Corpus level statistics (TFIDF, counts) were computed over the theories in the TheoryBank.
#
#
# ----
# Comparing TFIDF vs Unigram counts
#
# We use the mean reciprocal rank (MRR) for each vectorization strategy, along with whether 1-norm was applied or not.  Here, TFIDF without normalization was the clear winner.
#
# Count Norm: MRR=0.00126
# TFIDF Norm: MRR=0.00126
# Count NoNorm: MRR=0.00316
# TFIDF NoNorm: MRR=0.08397
#
# ----
# Running a SVC.  Note that this uses accuracy, and is not in a reranking form.


import collections
from collections import Counter
from pprint import pprint
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from pprint import pprint
from sklearn.model_selection import train_test_split
from IPython.display import JSON
from tqdm import tqdm
import numpy as np
import json
from enum import Enum
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from coprover import PROJ_ROOT, RSC_ROOT, PVSLIB_ROOT
from coprover.lemmaret.theorybank import TheoryBank, VecTypes, theory_files, gen_sbert_theorybank_notypes
from coprover.lemmaret.featurizer import *

DEBUG = False  # Activate this to make abbreviated versions for stress testing
print("Debug={}".format(DEBUG))
# Now go through each of the proofs and collect the lemma requests

LEMMA_OUTPUT_FPATH = Path(RSC_ROOT, "lemma_retrieval", "lemma_requests.json")
STATE = "state"
if DEBUG:
    RESULTS_ROOT = Path(PROJ_ROOT, "results", "debug")
else:
    RESULTS_ROOT = Path(PROJ_ROOT, "results")
RESULTS_ROOT.mkdir(exist_ok=True, parents=True)

# Load the lemma queries
print("Lemma queries cached, loading")
with open(LEMMA_OUTPUT_FPATH, 'r') as f:
    lemma_requests = json.load(f)

if DEBUG:
    lemma_requests = lemma_requests[0:100]

# Split into train/test
from sklearn.model_selection import train_test_split

lemma_train, lemma_test = train_test_split(
    lemma_requests, train_size=0.6, random_state=1337, shuffle=True, stratify=None)
print("# train={}, test={}, total={}".format(len(lemma_train), len(lemma_test), len(lemma_requests)))


sbert_noptypes_tb_norm = gen_sbert_theorybank_notypes()
experiments = [("SBert_NoType_Norm", sbert_noptypes_tb_norm)]

# Make several theory banks
if False:
    if not(DEBUG):
        count_tb_nonorm = TheoryBank(theory_files, vectorizer_type=VecTypes.COUNT, norm_vecs=False)
        tfidf_tb_nonorm = TheoryBank(theory_files, vectorizer_type=VecTypes.TFIDF, norm_vecs=False)
        sbert_tb_nonorm = TheoryBank(theory_files, vectorizer_type=VecTypes.SBERT, norm_vecs=False)

    count_tb_norm = TheoryBank(theory_files, vectorizer_type=VecTypes.COUNT, norm_vecs=True)
    tfidf_tb_norm = TheoryBank(theory_files, vectorizer_type=VecTypes.TFIDF, norm_vecs=True)
    sbert_tb_norm = TheoryBank(theory_files, vectorizer_type=VecTypes.SBERT, norm_vecs=True)

    if DEBUG:
        experiments = (("Count_Norm", count_tb_norm), ("TFIDF_Norm", tfidf_tb_norm),
                    ("SBert_Norm", sbert_tb_norm))
    else:
        experiments = (("Count_NoNorm", count_tb_nonorm), ("TFIDF_NoNorm", tfidf_tb_nonorm),
                    ("Count_Norm", count_tb_norm), ("TFIDF_Norm", tfidf_tb_norm),
                    ("SBert_NoNorm", sbert_tb_nonorm), ("SBert_Norm", sbert_tb_norm))


def make_statestr(state, consequents_only=False):
    """ Given lemma state, converts into query string usable for theory bank"""
    collecting = not(consequents_only)
    toks = []
    for tok in state:
        if tok == "consequents":
            collecting = True
        elif collecting:
            toks.append(str(tok))
    return " ".join(toks)


def compute_mrr(theory_bank):
    retrieval_ranks = []
    for req in tqdm(lemma_test):
        gold = req[NAME]
        if not (theory_bank.contains(gold)):
            continue
        state_str = make_statestr(req[STATE])
        retrieved = theory_bank.query(state_str, top_N=None)
        rank = retrieved.index(gold)
        retrieval_ranks.append(rank)
    mrr = np.mean([1 / (rank + 1) for rank in retrieval_ranks])
    return mrr, retrieval_ranks


MRR = "mrr"
NAME = "name"
RETRIEVAL_RANKS = "retrieval_ranks"
ACC_AT_N = "acc@{}"
results = []
for expname, tb in experiments:
    mrr, retrieval_ranks = compute_mrr(tb)
    print("{}:\t{:.5f}".format(expname, mrr))
    res_dict = {NAME: expname, MRR: mrr}
    for n in range(1,6):
        acc_at_n = len(np.where(np.array(retrieval_ranks) <= n)[0]) / len(retrieval_ranks)
        res_dict[ACC_AT_N.format(n)] = acc_at_n
        print("\tAcc@{}={:.2f}".format(n, acc_at_n))
    results.append(res_dict)
    with open(Path(RESULTS_ROOT, f"{expname}_retrieval_ranks.txt"), 'w') as f:
        for rank in retrieval_ranks:
            f.write(f"{rank}\n")

results_df = pd.DataFrame(results)
results_df.to_csv(Path(RESULTS_ROOT, "mrr_exps.csv"))

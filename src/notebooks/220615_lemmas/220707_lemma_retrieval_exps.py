#!/usr/bin/env python
# coding: utf-8

# In[1]:




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

# In[ ]:


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
from sklearn import svm

from coprover import PROJ_ROOT, RSC_ROOT, PVSLIB_ROOT
from featurizer import *

DATA_ROOT = Path(PROJ_ROOT, "data", "pvs", "pvslib")
json_files = list(DATA_ROOT.rglob("*.json"))

theory_files = list(DATA_ROOT.glob("*/*.json"))
proof_files = list(DATA_ROOT.glob("*/*/*.json"))
print(len(theory_files), len(proof_files))

import pdb

class VecTypes:
    COUNT = 1
    TFIDF = 2

class TheoryBank:
    def __init__(self, theory_files, 
                 vectorizer_type=VecTypes.COUNT,
                 norm_vecs=True):
        self.all_theories = collections.OrderedDict()
        self.all_lemmas = collections.OrderedDict()
        for json_fpath in tqdm(theory_files):
            with open(json_fpath, 'r') as f:
                theory_name = json_fpath.stem
                doc_root = json.load(f)
                theory = read_theory(doc_root)
                self.all_theories[theory_name] = theory
                self.all_lemmas.update(theory)
        corpus = [" ".join([str(y) for y in x]) for x in self.all_lemmas.values()]
        self.names = list(self.all_lemmas.keys())
        if vectorizer_type == VecTypes.COUNT:
            self.vectorizer = CountVectorizer(stop_words=None, lowercase=False)
        elif vectorizer_type == VecTypes.TFIDF:
            self.vectorizer = TfidfVectorizer(stop_words=None, lowercase=False)
        self.M = self.vectorizer.fit_transform(corpus).toarray()
        # 1-norm
        self.norm_vecs = norm_vecs
        if self.norm_vecs:
            self.M = self.M / np.linalg.norm(self.M, axis=1).reshape((self.M.shape[0], 1))
        
    def contains(self, name):
        return name in self.names
        
    def query(self, qdocs, top_N=5):
        """
        Expects a list of lists (docs x toks)
        """
        q = self.vectorizer.transform(qdocs).toarray()
        if self.norm_vecs:
            q = q / np.linalg.norm(q, axis=0)
        num_queries = len(qdocs)
        S = self.M.dot(q.transpose())
        sorted_idxes = np.argsort(S, axis=0)
        if top_N is None:
            max_idxes = sorted_idxes
        else:
            max_idxes = sorted_idxes[-top_N:, :]
        
        # Assemble list of names
        titles = []
        for qnum in range(len(qdocs)):
            titles.append([self.names[idx] for idx in max_idxes[::-1, qnum]])
        return titles
    
    def get(self, name, theories=None):
        if theories is None:
            theories = sorted(self.all_theories.keys())
        for theory in theories:
            if name in self.all_theories[theory]:
                return self.all_theories[theory][name], theory
        return None, None
        # raise Exception("Could not identify name={} in theory set={}".format(name, theories))
    
    def sample(self, rand_obj=None):
        if rand_obj is None:
            return np.random.choice(list(self.all_lemmas.values()))
        else:
            return rand_obj.choice(list(self.all_lemmas.values()))
        


# In[ ]:


theory_bank = TheoryBank(theory_files)


# In[ ]:


# Now go through each of the proofs and collect the lemma requests

LEMMA_OUTPUT_FPATH = Path("lemma_requests.json")
NAME = "name"
STEP = "step"
STATE = "state"
CMD = "command"
ARGS = "args"

if LEMMA_OUTPUT_FPATH.exists():
    # Load the lemma queries
    print("Lemma queries cached, loading")
    with open(LEMMA_OUTPUT_FPATH, 'r') as f:
        lemma_requests = json.load(f)
else:
    # Accumulate the lemma retrieval queries
    tqdm_iter = tqdm(proof_files)
    num_lemma_requests = 0
    lemma_requests = []


    for json_fpath in tqdm_iter:
        name = Path(json_fpath).stem
        sa_tuples = read_proof_session_lemmas(json_fpath)
        if sa_tuples is None:
            continue
        for step_num, sa_tuple in enumerate(sa_tuples):
            cmd, arg = sa_tuple[1], sa_tuple[2]
            if arg is not None and isinstance(arg, str):
                arg = arg.split("[")[0]
                if cmd in set(["lemma", "rewrite"]):
                    # expr, theory = theory_bank.get(arg)
                    num_lemma_requests += 1
                    lemma_requests.append({
                        STATE:  sa_tuple[0],
                        CMD: cmd,
                        ARGS: arg,
                        NAME: name,
                        STEP: step_num
                    })
                    tqdm_iter.set_description("# lemma requests={}".format(num_lemma_requests))

    # Save out the accumulated lemma requests to file                
    with open(LEMMA_OUTPUT_FPATH, 'w') as f:
        json.dump(lemma_requests, f)
        


# In[ ]:



# Split into train/test
from sklearn.model_selection import train_test_split
lemma_train, lemma_test = train_test_split(
     lemma_requests, train_size=0.6, random_state=1337, shuffle=True, stratify=None)
print("# train={}, test={}, total={}".format(len(lemma_train), len(lemma_test), len(lemma_requests)))


# In[ ]:


# Vectorize a query
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


# In[ ]:


# Make several theory banks
count_tb_nonorm = TheoryBank(theory_files, vectorizer_type=VecTypes.COUNT, norm_vecs=False)
tfidf_tb_nonorm = TheoryBank(theory_files, vectorizer_type=VecTypes.TFIDF, norm_vecs=False)
count_tb_norm = TheoryBank(theory_files, vectorizer_type=VecTypes.COUNT, norm_vecs=True)
tfidf_tb_norm = TheoryBank(theory_files, vectorizer_type=VecTypes.TFIDF, norm_vecs=True)

experiments = ( ("Count_NoNorm", count_tb_nonorm), ("TFIDF_NoNorm", tfidf_tb_nonorm),
              ("Count_Norm", count_tb_norm), ("TFIDF_Norm", tfidf_tb_norm))


# In[ ]:


def compute_mrr(theory_bank):
    retrieval_ranks = []
    for req in tqdm(lemma_test):
        gold = req[NAME]
        if not(theory_bank.contains(gold)):
            continue
        state_str = make_statestr(req[STATE])
        retrieved = theory_bank.query([state_str], top_N=None)
        rank = retrieved[0].index(gold)
        retrieval_ranks.append(rank)
    mrr = np.mean([1/(rank + 1) for rank in retrieval_ranks])
    return mrr


# In[21]:




# # Supervised experiment.  
# For each of the training items, set up a featurization that consists of q

# In[ ]:


from numpy.random import default_rng
import pdb
np_rng = default_rng(505)

def assemble_data(lemma_src, tb, limit=100):
    """
    Get one positive, exact match
    One easy negative, random sample
    One hard negative, random sample that is close
    """
    X = []
    Y = []
    for i in tqdm(range(limit)):
        req = lemma_src[i]
        state_str = make_statestr(req[STATE])
        q = tb.vectorizer.transform([state_str]).toarray()
        gold_lemma = tb.get(req[NAME])
        gold_x = tb.vectorizer.transform([make_statestr(gold_lemma)]).toarray()
        random_x = tb.vectorizer.transform([make_statestr(tb.sample())]).toarray()      
        x_pos = np.dot(q.transpose(), gold_x).ravel()
        x_neg = np.dot(q.transpose(), random_x).ravel()
        
        # Get the hard negative
        S = tb.M.dot(gold_x.transpose())
        sorted_idxes = np.argsort(S, axis=0)
        hard_neg_idx = np_rng.integers(low=1, high=len(sorted_idxes), size=1)[0]
        x_hard_neg = np.dot(q.transpose(), tb.M[hard_neg_idx].reshape((1, -1))).ravel()
        # pdb.set_trace()
        X.append(x_pos)
        X.append(x_neg)
        X.append(x_hard_neg)
        Y.append(1)
        Y.append(-1)
        Y.append(-1)        
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# In[ ]:


# import sklearn.svm as svm
# Try ThunderSVM, GPU accelerated SVC
# import thundersvm 


import collections
#num_train = 1000
num_train = 600
num_test = 60

svc_results = collections.OrderedDict()
for expname, theory_bank in experiments:
    svc = svm.LinearSVC()
    # svc = thundersvm.SVC()
    train_X, train_Y = assemble_data(lemma_train, theory_bank, num_train)
    test_X, test_Y = assemble_data(lemma_test, theory_bank, num_test)
    svc.fit(train_X, train_Y)
    train_Yhat = svc.predict(train_X)
    test_Yhat = svc.predict(test_X)
    train_acc = np.sum(train_Yhat == train_Y) / train_X.shape[0]
    test_acc = np.sum(test_Yhat == test_Y) / test_X.shape[0]
    svc_results[expname] = {
        "train_acc": train_acc,
        "test_acc": test_acc
    }
    print("{}: train_acc={:.3f}, test_acc={:.3f}".format(expname, train_acc, test_acc))


# In[ ]:





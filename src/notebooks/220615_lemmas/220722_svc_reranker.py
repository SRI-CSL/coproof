#!/usr/bin/env python
# coding: utf-8

# # SVC Reranking Experiment
# 
# Before we go full Transformer, let's see if we can use the classification result to act as a reranker instead.
# 
# We develop a probabilistic SVC via a multiplicative state difference over TFIDF representations.  We then train a RBF kernel SVC with probabilistic reranking activated.

# In[1]:


DEBUG = False


# In[2]:


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
import thundersvm

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
        return titles, max_idxes.ravel()
    
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
        

        
# Split into train/test
from sklearn.model_selection import train_test_split
lemma_train, lemma_test = train_test_split(
     lemma_requests, train_size=0.6, random_state=1337, shuffle=True, stratify=None)
print("# train={}, test={}, total={}".format(len(lemma_train), len(lemma_test), len(lemma_requests)))


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


# In[3]:


tfidf_tb_nonorm = TheoryBank(theory_files, vectorizer_type=VecTypes.TFIDF, norm_vecs=False)


# In[4]:


from numpy.random import default_rng
import pdb
np_rng = default_rng(505)

def featurize_mult(x1, x2):
    return np.multiply(x1, x2).ravel()

def featurize(x1, x2):
    return np.abs(x1 - x2).ravel()


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
        x_pos = featurize(q, gold_x)
        x_neg = featurize(q, random_x)
        
        # Get the hard negative
        S = tb.M.dot(gold_x.transpose())
        sorted_idxes = np.argsort(S, axis=0)
        hard_neg_idx = np_rng.integers(low=1, high=len(sorted_idxes), size=1)[0]
        x_hard_neg = featurize(q, tb.M[hard_neg_idx].reshape((1, -1)))
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


# In[5]:


from sklearn import svm
from pprint import pprint
from joblib import dump, load
# joblib is preferred for sklearn


#num_train = 1000
if DEBUG:
    num_train = 1000
    num_test = 100
    suffix = "debug"
else:
    num_train = len(lemma_train)
    num_test = len(lemma_test)
    suffix = "full"

print("Debug status={}".format(DEBUG))

theory_bank = tfidf_tb_nonorm
expname = "tfidf_nonorm"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_X, train_Y = assemble_data(lemma_train, theory_bank, num_train)
test_X, test_Y = assemble_data(lemma_test, theory_bank, num_test)

# svm_name = "RBF"
# svc = svm.SVC(kernel='rbf')

svm_name = "Poly"
kernel='poly'
svc = svm.SVC(kernel=kernel)

model_fpath = Path(MODEL_DIR, '{}.l1.{}.joblib'.format(kernel, suffix))
if model_fpath.exists():
    print("Model file={} exists, avoiding training".format(model_fpath))
    svc = load(model_fpath)
else:
    print("Training model, will save to {}".format(model_fpath))
    svc.fit(train_X, train_Y)
    print("... testing")
    train_Yhat = svc.predict(train_X)
    test_Yhat = svc.predict(test_X)
    train_acc = np.sum(train_Yhat == train_Y) / train_X.shape[0]
    test_acc = np.sum(test_Yhat == test_Y) / test_X.shape[0]
    model_result = {
        "exp_name": expname,
        "svc_type": svm_name,
        "train_acc": train_acc,
        "test_acc": test_acc
    }
    pprint(model_result)
    dump(svc, model_fpath)
    df = pd.DataFrame([model_result])
    df.to_csv(Path(MODEL_DIR, "{}.results.csv".format(model_fpath.stem)), header=True, index=False)


# In[52]:


# Now compute MRR, using the TFIDF to compute the initial rankings, and then to 
# rerank according to the SVC decision_function

theory_bank = tfidf_tb_nonorm
top_N=100
orig_retrieval_ranks = []
rerank_retrieval_ranks = []
tqdm_iter = tqdm(lemma_test)
for req in tqdm_iter:
    gold = req[NAME]
    if not(theory_bank.contains(gold)):
        continue
    state_str = make_statestr(req[STATE])
    retrieved_sets, retrieved_idxes = theory_bank.query([state_str], top_N=top_N)
    retrieved_titles = retrieved_sets[0]
    q = theory_bank.vectorizer.transform([state_str]).toarray()
    if gold not in retrieved_titles:
        continue
    orig_rank = retrieved_titles.index(gold)
    orig_retrieval_ranks.append(orig_rank)
    tqdm_iter.set_description("# processed = {}".format(len(orig_retrieval_ranks)))
    # Develop the subset of the test set corresponding to the retrieved idxes
    if True:
        candidates_X = theory_bank.M[retrieved_idxes]
        candidate_scores = svc.decision_function(candidates_X)
        sorted_idxes = np.argsort(candidate_scores)
        sorted_titles = np.array(retrieved_titles)[sorted_idxes[::-1]]
        sorted_rank = list(sorted_titles).index(gold)
        rerank_retrieval_ranks.append(sorted_rank)


# In[53]:


orig_mrr = np.mean([1/(rank + 1) for rank in orig_retrieval_ranks])
rerank_mrr = np.mean([1/(rank + 1) for rank in rerank_retrieval_ranks])
print("Orig MRR={:.5f}, Rerank MRR={:.5f}".format(orig_mrr, rerank_mrr))


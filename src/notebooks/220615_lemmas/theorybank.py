""" 
Library for accessing saved theories

Main entrypoint is gen_default_theorybank(), for assembling a TheoryBank.

Use assemble_lemma_requests() to create the lemma_requests.json file, which lists all of the lemma requests.
"""

import collections
from collections import Counter
from pprint import pprint
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from pprint import pprint
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import json
from enum import Enum
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import thundersvm

from coprover import PROJ_ROOT, RSC_ROOT, PVSLIB_ROOT
from featurizer import *


PVSLIB_ROOT = Path(PROJ_ROOT, "data", "pvs", "pvslib")
PRELUDE_ROOT = Path(PROJ_ROOT, "data", "pvs", "prelude")

theory_files = []
proof_files = []

# Iterate through all of the theories in PVSLib and accumulate the proof and theory files
for theory_dir in PVSLIB_ROOT.glob("*/"):
    theory_files.extend(theory_dir.glob("*.json"))
    proof_files.extend(theory_dir.glob("*/*.json"))

# Now go through the PVS Prelude, but only collect theories, as proofs are not representative
# of actual use.
theory_files.extend(PRELUDE_ROOT.glob("*.json"))
    
print(len(theory_files), len(proof_files))

import pdb

def gen_default_theorybank():
    return TheoryBank(theory_files)

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
    
    def __len__(self):
        return len(self.names)
   
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
        
#
# Code for constructing the lemma_requests cache
#
        
LEMMA_OUTPUT_FPATH = Path("lemma_requests.json")
NAME = "name"
STEP = "step"
STATE = "state"
CMD = "command"
ARGS = "args"

def assemble_lemma_requests(proof_files):
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
        
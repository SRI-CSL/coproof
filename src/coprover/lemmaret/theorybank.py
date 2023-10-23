"""
Library for accessing saved theories

Main entrypoint is gen_default_theorybank(), for assembling a TheoryBank.

Use assemble_lemma_requests() to create the lemma_requests.json file, which lists all of the lemma requests.
"""

import collections
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from coprover import PROJ_ROOT, RSC_ROOT, PVSLIB_ROOT
from coprover.utils import ensure_numpy
from .vectorizers import SBertVectorizer
from .featurizer import *

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
    """ Constructs a TheoryBank that uses the PVS Prelude and PVSLib theories."""
    return TheoryBank(theory_files)

def gen_count_theorybank():
    return TheoryBank(theory_files, vectorizer_type=VecTypes.COUNT)

def gen_tfidf_theorybank():
    return TheoryBank(theory_files, vectorizer_type=VecTypes.TFIDF)

def gen_sbert_theorybank():
    """ Constructs a Theorybank with a trained SBert vectorizer"""
    return TheoryBank(theory_files, vectorizer_type=VecTypes.SBERT, norm_vecs=True)


def gen_sbert_theorybank_notypes():
    """ Constructs a Theorybank with a trained SBert vectorizer, all generic types"""
    return TheoryBank(theory_files, vectorizer_type=VecTypes.SBERT_NOTYPES, norm_vecs=True)

class VecTypes:
    NONE = 0
    COUNT = 1
    TFIDF = 2
    SBERT = 3  # Fitted and trained SBert
    SBERT_NOTYPES = 4  # Fitted and trained SBert, no types

class TheoryBank:
    def __init__(self, theory_files,
                 vectorizer_type=VecTypes.COUNT,
                 cache_dir=Path(RSC_ROOT, "lemma_retrieval", "vec_cache"),
                 norm_vecs=False):
        cache_dir.mkdir(exist_ok=True, parents=True)
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
        if vectorizer_type == VecTypes.NONE:
            self.vectorizer = None
            self.cache_fpath = None
        elif vectorizer_type == VecTypes.COUNT:
            self.vectorizer = CountVectorizer(stop_words=None, lowercase=False)
            self.cache_fpath = Path(cache_dir, "count.npy")
        elif vectorizer_type == VecTypes.TFIDF:
            self.vectorizer = TfidfVectorizer(stop_words=None, lowercase=False)
            self.cache_fpath = Path(cache_dir, "tfidf.npy")
        elif vectorizer_type == VecTypes.SBERT:
            self.vectorizer = SBertVectorizer()
            self.cache_fpath = Path(cache_dir, "sbert.npy")
        elif vectorizer_type == VecTypes.SBERT_NOTYPES:
            self.vectorizer = SBertVectorizer(use_var_types=False)
            self.cache_fpath = Path(cache_dir, "sbert_notypes.npy")    

        if self.cache_fpath.exists():
            print("Loading TheoryBank vector cache from {}".format(self.cache_fpath))
            self.vectorizer.fit(corpus)
            with open(self.cache_fpath, "rb") as f:
                self.M = np.load(f)
        elif vectorizer_type == VecTypes.NONE:
            print("No vectorization desired, skipping")
        else:
            print("Vectorizing corpus")
            self.M = ensure_numpy(self.vectorizer.fit_transform(corpus))
            print("Saving cache out to {}".format(self.cache_fpath))
            np.save(self.cache_fpath, self.M)
        # 1-norm
        self.norm_vecs = norm_vecs
        if self.norm_vecs and not(vectorizer_type == VecTypes.NONE):
            self.M = self.M / np.linalg.norm(self.M, axis=1).reshape((self.M.shape[0], 1))

    def __len__(self):
        return len(self.names)

    def contains(self, name):
        return name in self.names

    def vectorize(self, query_text):
        qdocs = [query_text]
        q = ensure_numpy(self.vectorizer.transform(qdocs))
        if self.norm_vecs:
            q = q / np.linalg.norm(q, axis=1)
        return q

    def query(self, query_text, top_N=5, return_scores=False):
        """
        Expects a single document (text) for now.  Batched
        versions in the future
        """
        assert isinstance(query_text, str)
        qdocs = [query_text]
        q = ensure_numpy(self.vectorizer.transform(qdocs))
        if self.norm_vecs:
            q = q / np.linalg.norm(q, axis=1)
        num_queries = len(qdocs)
        S = self.M.dot(q.transpose())
        # For now, only work with a single query
        sorted_idxes = np.argsort(S[:,0])
        sorted_scores = S[sorted_idxes]
        if top_N is None:
            max_idxes = sorted_idxes
        else:
            sorted_scores = sorted_scores[-top_N:]
            max_idxes = sorted_idxes[-top_N:]

        # Assemble list of names, reverse so most relevant first
        titles = [self.names[idx] for idx in max_idxes][::-1]
        if return_scores:
            sorted_scores = sorted_scores[::-1]
            ret = []
            for title, score in zip(titles, sorted_scores):
                ret.append((title, float(score)))
            return ret
        else:
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


NAME = "name"
STEP = "step"
STATE = "state"
CMD = "command"
ARGS = "args"


def assemble_lemma_requests(proof_files,
                            lemma_output_fpath=Path("lemma_requests.json")):
    if lemma_output_fpath.exists():
        # Load the lemma queries
        print("Lemma queries cached, loading")
        with open(lemma_output_fpath, 'r') as f:
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
                            STATE: sa_tuple[0],
                            CMD: cmd,
                            ARGS: arg,
                            NAME: name,
                            STEP: step_num
                        })
                        tqdm_iter.set_description("# lemma requests={}".format(num_lemma_requests))

        # Save out the accumulated lemma requests to file
        with open(lemma_output_fpath, 'w') as f:
            json.dump(lemma_requests, f)

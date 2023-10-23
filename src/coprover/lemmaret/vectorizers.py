"""
Vectorizers for populating vectors
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

from coprover import RSC_ROOT
from coprover.utils import ensure_numpy
from .featurizer import process_formula_json

class LemmaVectorizer:
    def vectorize(self, formula) -> np.ndarray:
        pass

    def fit(self, corpus):
        pass

    def fit_transform(self, corpus) -> np.ndarray:
        pass


class SBertVectorizer(LemmaVectorizer):
    def __init__(self, model_fpath=Path(RSC_ROOT, "lemma_retrieval", "nir_sbert"),
                 featurize=False, verbose=False, use_var_types=True):
        super(LemmaVectorizer, self).__init__()
        self.model = SentenceTransformer(str(model_fpath))
        self.featurize = featurize
        self.verbose = verbose
        self.use_var_types = use_var_types

    def vectorize(self, formula) -> np.ndarray:
        if self.featurize:
            formula = process_formula_json(formula, process_types=self.use_var_types)
        x = ensure_numpy(self.model.encode(formula))
        return x

    def fit(self, corpus):
        pass

    def transform(self, corpus) -> np.ndarray:
        """ Note this is technically just a transform, as the model has already been fitted."""
        X = []
        if self.verbose:
            corpus_iter = tqdm(corpus)
        else:
            corpus_iter = iter(corpus)
        for lemma in corpus_iter:
            x = self.vectorize(lemma)
            X.append(x)
        return np.array(X)

    def fit_transform(self, corpus) -> np.ndarray:
        return self.transform(corpus)
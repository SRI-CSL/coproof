"""
Sets up the queries and splits used for comparative experiments
"""

import json
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from .theorybank import gen_default_theorybank, assemble_lemma_requests, proof_files

LEMMA_PREFIX = "LEMMA"
STATE_PREFIX = "STATE"

BATCH_SIZE = 4

#
# This reads in the requests, divides them according to the train/test partitions established in
# 220707_lemma_retrieval_exps.


# Read in Lemma requests
assemble_lemma_requests(proof_files)

with open("lemma_requests.json", "r") as f:
    lemma_requests = json.load(f)

theorybank = gen_default_theorybank()

# Split into train/test
train_queries, test_queries = train_test_split(
    lemma_requests, train_size=0.6, random_state=1337, shuffle=True, stratify=None)
print("# train={}, test={}, total={}".format(len(train_queries), len(test_queries), len(lemma_requests)))

all_lemmas = list(theorybank.all_lemmas.values())
all_lemma_names = list(theorybank.all_lemmas.keys())


def assemble(queries, all_lemmas, total_pairs, add_prefix_type=True):
    """
    Assemble the queries and target lemmas, along with negative lemmas.
    If prefix_type is True, we prefix lemmas with LEMMA_PREFIX and state with STATE_PREFIX
    to condition these.
    """
    accum_examples = []
    while len(accum_examples) < total_pairs:
        i1 = np.random.randint(len(queries))
        lr = queries[i1]
        gold_lemma = lr['name']
        if theorybank.contains(gold_lemma):
            state = " ".join([str(x) for x in lr['state']])
            if add_prefix_type:
                state = STATE_PREFIX + " " + state
            gold_lemma_body = theorybank.get(gold_lemma)[0]
            if gold_lemma_body is not None:
                lemma_body = " ".join([str(x) for x in gold_lemma_body])
                if add_prefix_type:
                    lemma_body = LEMMA_PREFIX + " " + lemma_body
                # Sample random lemma
                neg_lemma_body = lemma_body
                while neg_lemma_body == lemma_body:
                    idx = np.random.randint(len(all_lemmas))
                    neg_title = all_lemma_names[idx]
                    neg_lemma_body = " ".join([str(x) for x in all_lemmas[idx]])
                pos_example = InputExample(texts=[state, lemma_body], label=1.0)
                neg_example = InputExample(texts=[state, neg_lemma_body], label=0.0)
                pos_example.lemma_name = gold_lemma
                neg_example.lemma_name = neg_title
                accum_examples.append(pos_example)
                accum_examples.append(neg_example)
                if len(accum_examples) % (total_pairs // 10) == 0:
                    print("{}/{}".format(len(accum_examples), total_pairs))
    return accum_examples


def setup_data(add_prefix_type=False):
    train_examples = assemble(train_queries, all_lemmas, 100_000, add_prefix_type=add_prefix_type)
    test_examples = assemble(test_queries, all_lemmas, 100, add_prefix_type=add_prefix_type)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    return train_examples, test_examples, train_dataloader

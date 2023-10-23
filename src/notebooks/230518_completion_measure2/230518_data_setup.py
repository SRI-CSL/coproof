#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Sets up the data for predicting the end token
"""

from tqdm import tqdm

from coprover.lemmaret.vectorizers import *


# In[2]:


sbv = SBertVectorizer()


# In[3]:


sbv.vectorize("<ANT> <CONS> s-formula apply constant apply constant constant constant <HID>").shape


# In[4]:


tokenizer = sbv.model.tokenizer


# In[5]:


from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
)


# In[6]:


tokenizer.encode("<ANT> <CONS> s-formula apply constant apply constant constant constant <HID>")


# In[7]:


# Now msuter the data
from pathlib import Path
import pandas as pd

from coprover import DATA_ROOT, RSC_ROOT

cmdpred_df = pd.read_csv(Path(RSC_ROOT, "pvs_cmd_pred", "data","cmdpred_N3.pvslib.tsv.gz"), sep='\t', names=['sequent', 'command', 'cmd_history', 'uri'])
                         


# In[8]:


cmdpred_df[0:22]


# In[9]:


# Identify proofs by name
start_state = None
last_state = None
last_proofname = None

def get_proofname(uri):
    return uri.split('#', 1)[0]


proofnames = [get_proofname(row.uri) for idx, row in cmdpred_df.iterrows()]
cmdpred_df['proofname'] = proofnames
grp_obj = cmdpred_df.groupby('proofname')
proofnames = list(grp_obj.groups.keys())
print(f"Total unique proofs={len(proofnames)}")


# In[10]:


# Identify lengths of proofs
proof_lengths = []
for proofname in tqdm(proofnames):
    rows = grp_obj.get_group(proofname)
    proof_lengths.append(len(rows))

from collections import Counter
from pprint import pprint
print("Histogram of proof lengths")
pprint(Counter(proof_lengths))


# In[15]:


PROOFNAME = "proofname"
STATE = "source_text"
LABEL = "target_text"
POS = "{}"  # Formatted label for number of steps to end
NEG = "neg"
CMD_HISTORY = "cmd_history"

class MTuple:
    def __init__(self, proofname, start_row, curr_row, label):
        self.proofname = proofname
        self.label = label
        self.start_row, self.curr_row = start_row, curr_row
        self.cmd_history = self.curr_row.cmd_history
        assert str(self.start_row) != str(self.curr_row)

    def __str__(self, str):
        return self.proofname
    
    def _statestr(self):
        cmdhist_str = self.curr_row.cmd_history.replace(",", " ")
        return "{} {} {}".format(cmdhist_str, self.start_row.sequent, self.curr_row.sequent)  # First naive formulation
    
    def as_row(self):
        return {
            PROOFNAME: self.proofname,
            STATE: self._statestr(),
            CMD_HISTORY: self.curr_row.cmd_history.replace(",", " "),
            LABEL: self.label
        }


# In[21]:


from random import Random

rnd = Random()
rnd.seed(1337)

pos_mtuples = []
neg_mtuples = []
hard_neg_mtuples = []

N_END = 2 # Up to 2 from end
NUM_NEGATIVES = N_END + 1 
for proofname in tqdm(proofnames):
    rows = grp_obj.get_group(proofname)
    if len(rows) >= (2 + N_END + 1):  # Enough for separate start and end, with padding, and a hard negative
        start_row = rows.iloc[0]
        for n_idx in range(0, N_END + 1):
            curr_row = rows.iloc[ len(rows) - (n_idx + 1)]
            label = POS.format(n_idx)
            pos_mtuples.append(MTuple(proofname, start_row, curr_row, label))
        hard_neg_row = rows.iloc[len(rows) - (N_END + 1)]  # Hard negative, the one right before
        hard_neg_mtuples.append(MTuple(proofname, start_row, hard_neg_row, NEG))
        # Sample N negatives
        for _ in range(NUM_NEGATIVES):
            random_neg_row = rows.iloc[rnd.randint(1, len(rows) - N_END)]
            neg_mtuples.append(MTuple(proofname, start_row, random_neg_row, NEG))


# In[22]:


idx=0
print("Start")
print(pos_mtuples[idx].start_row.sequent, pos_mtuples[idx].start_row.cmd_history)
print("\n\nPositive")
print(pos_mtuples[idx].curr_row.sequent, pos_mtuples[idx].curr_row.cmd_history)

print("\n\nNegative")
print(neg_mtuples[idx].curr_row.sequent, neg_mtuples[idx].curr_row.cmd_history)

print("\n\nHard Negative")
print(hard_neg_mtuples[idx].curr_row.sequent, hard_neg_mtuples[idx].curr_row.cmd_history)


# In[23]:


# save as a dataframe and then feed into simple_t5
# setup so only tuples below max token lengths can be used

filtered_rows = []
total = 0
for mt in pos_mtuples + neg_mtuples:
    entry = mt.as_row()
    total += 1
    if len(entry[STATE].split()) <= 1000:
        filtered_rows.append(entry)

print("Filtered size={}/{}".format(len(filtered_rows), total))
inst_df = pd.DataFrame(filtered_rows)
inst_df.to_csv("laststep_pred.v2.csv.gz", header=True)


# In[24]:


inst_df


# In[26]:


inst_df['target_text'].hist()


#!/usr/bin/env python
# coding: utf-8

# This sets up a SBert instance using the Roberta model trained with 220818_train_llm and 220817_create_corpus
# 
# We develop a corpus from the train/test lemmas and queries.
# 
# Note that currently we seem to need to use a pre-trained BERT model for the training to work properly, as the custom vocabulary and pre-trained objective does not seem to be working well

# In[1]:


from pathlib import Path
from sentence_transformers import SentenceTransformer, models
from transformers import RobertaTokenizerFast, AutoModel
from setup_queries import *


# In[2]:


DOC_LIM=859
MODEL_SAVE_DIR = Path("outputs/nir_sbert/")
MODEL_OUTPUT_PATH = Path(MODEL_SAVE_DIR, "outputs")


# In[3]:


tokenizer = RobertaTokenizerFast.from_pretrained("outputs/nir_vocab", max_len=DOC_LIM)
vocab_size = len(tokenizer)


# In[4]:


WORD_EMBED_DIM=768 # Obtained by model summary
import sentence_transformers
# word_embedding_model = AutoModel.from_pretrained("outputs/nir_model/checkpoint-1100000")
word_embedding_model = sentence_transformers.models.Transformer("outputs/nir_model/checkpoint-1100000",
                                                               tokenizer_name_or_path="outputs/nir_vocab",
                                                               max_seq_length=DOC_LIM)
pooling_model = models.Pooling(WORD_EMBED_DIM)
# word_embedding_model.tokenizer = tokenizer.tokenize # Set the function
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Now set up a set of SentBERT InputExample entries for contrastive learning
# We first generate a sample set from the set of theories and queries.  We first
# train a pure Siamese similarity function, and then assess the ranking ability of
# that against queries.  We then train a model specifically to do the asymmetric similarity assessment.
# 
# First we use the theory bank and assemble an in-memory dataset 

# In[5]:


import numpy as np
np.random.seed(1337)
np.random.randint(100)


# In[7]:


# Read in Lemma requests
import json
with open("lemma_requests.json", "r") as f:
    lemma_requests = json.load(f)
    
    
# Split into train/test
from sklearn.model_selection import train_test_split
train_queries, test_queries = train_test_split(
     lemma_requests, train_size=0.6, random_state=1337, shuffle=True, stratify=None)
print("# train={}, test={}, total={}".format(len(train_queries), len(test_queries), len(lemma_requests)))


# In[8]:


from theorybank import gen_default_theorybank

theorybank = gen_default_theorybank()


# In[9]:


# TODO: Load the lemma requests.json, and use the lemma query and the lemma body in library as true positives.

from tqdm import tqdm
from sentence_transformers import InputExample, SentencesDataset
from torch.utils.data import DataLoader
import numpy as np

from setup_queries import *


# In[19]:


# Set up the evaluator
eval_set = test_examples
sentences1 = [x.texts[0] for x in eval_set]
sentences2 = [x.texts[1] for x in eval_set]
labels = [x.label for x in eval_set]
evaluator = sentence_transformers.evaluation.BinaryClassificationEvaluator(sentences1, sentences2, labels)


# In[20]:


def status(score, epoch, steps):
    print("Score={:.5f}, epoch={}, steps={}".format(score, epoch, steps))


# In[21]:


from sentence_transformers import losses

train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1000, warmup_steps=100,
          evaluator=evaluator,
          output_path=str(MODEL_OUTPUT_PATH), 
          callback=status,
         checkpoint_path=str(MODEL_SAVE_DIR), checkpoint_save_steps=10000, checkpoint_save_total_limit=5)


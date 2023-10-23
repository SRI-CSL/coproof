#!/usr/bin/env python
# coding: utf-8

# This sets up a SBert instance using the Roberta model trained with 220818_train_llm and 220817_create_corpus
#
# We develop a corpus from the train/test lemmas and queries.
#
# This follows 220818a_sbert's training protocol, but does an analysis every
#
# Note that currently we seem to need to use a pre-trained BERT model for the training to work properly, as the custom vocabulary and pre-trained objective does not seem to be working well


"""
Executables for training underlying models.  This is a concatenation of two notebooks in 220615_lemmas/,

- 220817_create_corpus.ipynb (vocabulary training)
- 220818_train_llm.ipynb (training constituent LLM)

"""
import json
from pathlib import Path
from collections import Counter
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer

from coprover.lemmaret.featurizer import *
from coprover.lemmaret.theorybank import *


#
# Training Vocabulary
#

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

# Set up for training a vocabulary
# Assemble a corpus containing all of the lemmas, and then all of the sequents where the lemma requests are made
# NOTE: May want to save commands as bpe encoded tokens, given new commands may be coded up.

with open("lemma_requests.json", "r") as f:
    lemma_requests = json.load(f)

theorybank = gen_default_theorybank()

corpus_fpath = Path(OUTPUTS_DIR, "corpus.txt")
commands_fpath = Path(OUTPUTS_DIR, "commands.txt")
with open(corpus_fpath, 'w') as f:
    with open(commands_fpath, 'w') as cmd_f:
        for name, lemma_body in theorybank.all_lemmas.items():
            f.write(" ".join([str(x) for x in lemma_body]))
            f.write("\n")
        for lreq in lemma_requests:
            f.write(" ".join([str(x) for x in lreq['state']]))
            f.write("\n")
            cmd_f.write("{}\n".format(lreq['command']))

# Now train the vocabulary model



theorybank = gen_default_theorybank()
from coprover import RSC_ROOT

Path("outputs/nir_vocab").mkdir(exist_ok=True, parents=True)

# Special tokens used by sequent representation and for Roberta model
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SEP_TOKEN = "<sep>"
CLS_TOKEN = "<cls>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"

SPEC_TOKS=[
    "antecedent",
    "consequent",
    "hidden",
    "null",
    BOS_TOKEN,
    EOS_TOKEN,
    SEP_TOKEN,
    CLS_TOKEN,
    UNK_TOKEN,
    PAD_TOKEN,
    MASK_TOKEN
    ]

corpus_files = [Path(OUTPUTS_DIR, "corpus.txt")]

# Get vocab histogram
tok_freq = Counter()
for fpath in corpus_files:
    with open(fpath, 'r') as f:
        for line in f:
            toks = line.strip().split()
            tok_freq.update(toks)

vocab_size = len(tok_freq)
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=[str(x) for x in corpus_files], vocab_size=vocab_size, min_frequency=1, special_tokens=SPEC_TOKS)

# NOTE: Need to save_model and then save for all the necessary files.
tokenizer.save_model("outputs/nir_vocab")
tokenizer.save("outputs/nir_vocab/config.json")


#
# This now trains the RoBERTa model on the lemma and formulas, which will be used as the base formula encoder for
# the Siamese network
#

from pathlib import Path
import numpy as np
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM

corpus_file = Path("outputs/corpus.txt")

# Analyze the corpus file, stats on number of tokens
tok_lengths = []
with open(corpus_file, 'r') as f:
    for line in f:
        toks = line.split()
        tok_lengths.append(len(toks))
max_toks, mean_toks, std_toks = np.max(tok_lengths), np.mean(tok_lengths), np.std(tok_lengths)
print(max_toks, mean_toks, std_toks)

DOC_LIM = int(mean_toks + 2 * std_toks)

print(DOC_LIM)
tokenizer = RobertaTokenizerFast.from_pretrained("outputs/nir_vocab", max_len=DOC_LIM)
vocab_size = len(tokenizer)
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=corpus_file,
    block_size=DOC_LIM,  # For each line, truncates to this size.  For one doc per line, this should be limit on encoder
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

model_output_dir = Path("outputs/encoder_model")
model_output_dir.mkdir(exist_ok=True, parents=True)


config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=DOC_LIM + 2,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
model = RobertaForMaskedLM(config=config)

training_args = TrainingArguments(
    output_dir=model_output_dir,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()

#
# Now train the sentence-bert on the given vocab and embedder
#

# In[1]:


from pathlib import Path
from sentence_transformers import SentenceTransformer, models
from transformers import RobertaTokenizerFast, AutoModel
from coprover.lemmaret.setup_exp_queries import *

train_examples, test_examples, train_dataloader = setup_data(add_prefix_type=False)

# In[2]:


DOC_LIM = 859
MODEL_SAVE_DIR = Path("outputs/nir_sbert/")
MODEL_SAVE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_OUTPUT_PATH = Path(MODEL_SAVE_DIR, "outputs")

# In[3]:


tokenizer = RobertaTokenizerFast.from_pretrained("outputs/nir_vocab", max_len=DOC_LIM)
vocab_size = len(tokenizer)

# In[4]:


WORD_EMBED_DIM = 768  # Obtained by model summary
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
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=100,
          evaluator=evaluator,
          output_path=str(MODEL_OUTPUT_PATH),
          callback=status,
          checkpoint_path=str(MODEL_SAVE_DIR), checkpoint_save_steps=1000, checkpoint_save_total_limit=1000)


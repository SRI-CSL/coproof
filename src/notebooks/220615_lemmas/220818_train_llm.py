#!/usr/bin/env python
# coding: utf-8

# Trains the LLM, using the corpus and vocabularies created in 220817_create_corpus.ipynb

# In[1]:


from pathlib import Path
import numpy as np
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM


# In[2]:


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


# In[3]:


tokenizer = RobertaTokenizerFast.from_pretrained("outputs/nir_vocab", max_len=DOC_LIM)


# In[4]:


vocab_size = len(tokenizer)


# In[5]:


dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=corpus_file,
    block_size=DOC_LIM,  # For each line, truncates to this size.  For one doc per line, this should be limit on encoder
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


# In[6]:


model_output_dir = Path("outputs/nir_model")
model_output_dir.mkdir(exist_ok=True, parents=True)


# In[7]:



config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=DOC_LIM + 2, 
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
model = RobertaForMaskedLM(config=config)


# In[8]:


training_args = TrainingArguments(
    output_dir=model_output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1000,
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


# In[9]:


trainer.train()


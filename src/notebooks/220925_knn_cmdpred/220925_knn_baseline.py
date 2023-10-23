#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Applies simplet5 setup, but uses a KNN classifier.

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

from coprover import RSC_ROOT


# In[2]:


USE_MLM = False

# Command prefixes are always expected
CMD1_PREFIX = "command1: "

SRC_TXT = 'source_text'
TGT_TXT = 'target_text'
CMD_HIST = 'cmd_history'
BRANCH = 'branch'
DEPTH = 'depth'

#DATA_FPATH = Path(RSC_ROOT, "pvs_cmd_pred", "data", "cmdpred_N3.prelude.tsv.gz")
DATA_FPATH = Path(RSC_ROOT, "pvs_cmd_pred", "data", "cmdpred_N3.pvslib.tsv.gz")

full_df = pd.read_csv(DATA_FPATH,
                      sep="\t",
                      header=None,
                      names=[SRC_TXT,
                             TGT_TXT,
                             CMD_HIST,
                             BRANCH,
                             DEPTH])

# Subsample the DF to limit train/test times
full_df = full_df.sample(n=20000, random_state=42)

# Use full command history, with cmdhist as a single tok 
full_df['source_text'] = CMD1_PREFIX + full_df[CMD_HIST].replace(",", "") + " <pad> " + full_df[SRC_TXT]

# Try without command history
# full_df['source_text'] = CMD1_PREFIX  + " <pad> " + full_df[SRC_TXT]

# Try just with command history
# full_df['source_text'] = CMD1_PREFIX + full_df[CMD_HIST].replace(",", "")

train_df, test_df = train_test_split(full_df, test_size=0.1,
                                     random_state=1337,
                                     shuffle=True)


# In[3]:


train_df


# In[4]:


train_corpus = []
train_targets = []
for row in train_df.iterrows():
    src_txt = row[1][SRC_TXT]
    cmd = row[1][TGT_TXT]
    train_corpus.append(src_txt)
    train_targets.append(cmd)


# In[5]:


test_corpus = []
test_targets = []
for row in test_df.iterrows():
    src_txt = row[1][SRC_TXT]
    cmd = row[1][TGT_TXT]
    test_corpus.append(src_txt)
    test_targets.append(cmd)


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import collections

src_txt_vectorizer = TfidfVectorizer(stop_words=None)
label_lookup = collections.OrderedDict()

train_X = src_txt_vectorizer.fit_transform(train_corpus).toarray()
test_X = src_txt_vectorizer.transform(test_corpus).toarray()

# Combine train and test targets, so we get full coverage of labels
for target in train_targets + test_targets:
    if target not in label_lookup:
        label_lookup[target] = len(label_lookup)

def transform_labels(targets):
    Y = np.array([label_lookup[target] for target in targets])
    return Y

train_Y = transform_labels(train_targets)
test_Y = transform_labels(test_targets)

# Now perform the sanity check
for idx in range(len(train_targets)):
    label = train_targets[idx]
    label_id = train_Y[idx]
    assert(label_lookup[label] == label_id)


from sklearn.metrics import classification_report

def get_acc(Y, Yhat):
    acc = np.sum(Y == Yhat) / len(Y)
    return acc
    
def assess(Y, Yhat):
    acc = np.sum(Y == Yhat) / len(Y)
    print("Acc = {:.5f}".format(acc))
    print(classification_report(Y, Yhat))
    return acc


# In[ ]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


clf = make_pipeline(StandardScaler(), SVC(kernel='poly', random_state=0, tol=1e-5))
clf.fit(train_X, train_Y)
train_Yhat = clf.predict(train_X)
test_Yhat = clf.predict(test_X)

print("- - - - -\n SVC Poly")
print("Training")
assess(train_Y, train_Yhat)

print("Test")
assess(test_Y, test_Yhat)


# In[ ]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=0, tol=1e-5))
clf.fit(train_X, train_Y)
train_Yhat = clf.predict(train_X)
test_Yhat = clf.predict(test_X)

print("- - - - -\n SVC RBF")
print("Training")
assess(train_Y, train_Yhat)

print("Test")
assess(test_Y, test_Yhat)


# In[ ]:


from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
clf.fit(train_X, train_Y)
train_Yhat = clf.predict(train_X)
test_Yhat = clf.predict(test_X)

print("- - - - -\n LinearSVC")
print("Training")
assess(train_Y, train_Yhat)

print("Test")
assess(test_Y, test_Yhat)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


for N in range(1,11):
    knn_clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=N, weights='distance'))
    knn_clf.fit(train_X, train_Y)
    train_Yhat = knn_clf.predict(train_X)
    test_Yhat = knn_clf.predict(test_X)
    print("- - - - -\nkNN N={}".format(N))
    print("Training")
    print("{:.5f}".format(get_acc(train_Y, train_Yhat)))
    print("Test")
    print("{:.5f}".format(get_acc(test_Y, test_Yhat)))


"""
Trial LazyPredict
"""
from pathlib import Path
from lazypredict.Supervised import LazyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

NGRAM_RANGE=(1,3)
prefix="uni_bi_tri"

src_txt_vectorizer = TfidfVectorizer(stop_words=None, ngram_range=NGRAM_RANGE)
from coprover.cmdpred.cmdpred_data_setup import get_full_dfs, convert2XY
train_df, test_df = get_full_dfs()

train_X, train_Y, test_X, test_Y = convert2XY(train_df, test_df, src_txt_vectorizer=src_txt_vectorizer)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None, predictions=True)
models, predictions = clf.fit(train_X, test_X, train_Y, test_Y)

print(models)

res_dir = Path("results")
res_dir.mkdir(ensure_parents=True, exist_ok=True)

predictions.to_csv(Path(res_dir, "{}.lazypred.predictions.csv".format(prefix)))
models.to_csv(Path(res_dir, "{}.lazypred.models.csv".format(prefix)))

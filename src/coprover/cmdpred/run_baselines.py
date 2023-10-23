import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from coprover import RSC_ROOT
from coprover.results_analysis import *
from coprover.cmdpred.cmdpred_data_setup import *
from coprover.cmdpred import CmdPredictor

DEBUG = False

print("DEBUG={}".format(DEBUG))

EXPERIMENTS = (
    ("GradientBoosted", make_pipeline(StandardScaler(), GradientBoostingClassifier())),
    ("LinearSVC", make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))),
    ("SVM_RBF",  make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=0, tol=1e-5))),
    ("KNN_n=5_dist", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance')))
)

NGRAM_RANGE = (1, 3)
NGRAM_RANGE_PREFIX = "ngrams={}".format(NGRAM_RANGE)

for EXP_TYPE in (FULL, SEQONLY, CMDHISTONLY):
    print("Running exptype={}".format(EXP_TYPE))
    RES_DIR = Path("results")
    RES_DIR.mkdir(exist_ok=True, parents=True)

    train_df, test_df = get_splits(exp_type=EXP_TYPE)
    vectorizer = TfidfVectorizer(stop_words=None, ngram_range=NGRAM_RANGE)
    train_X, train_Y, test_X, test_Y, label_lookup = convert2XY(train_df, test_df,
                                                                src_txt_vectorizer=vectorizer)
    if DEBUG:
        train_X = train_X[0:1000]
        train_Y = train_Y[0:1000]
    ordered_label_lookup = []
    labels = label_lookup.items()
    sorted(labels, key=lambda x:x[1]) # Sort by the assigned idx
    print("Labels:\n{}".format(labels))
    labels = [x[0] for x in labels]

    tqdm_iter = tqdm(EXPERIMENTS)
    for x in tqdm_iter:
        exp_name, exp_model = x[0], x[1]
        tqdm_iter.set_description(("Exp={}.{}".format(exp_name, NGRAM_RANGE_PREFIX)))
        exp_model.fit(train_X, train_Y)
        guess_train_Y = exp_model.predict(train_X)
        train_telemetry = GuessGoldTelemetry(guess_train_Y, train_Y)
        train_telemetry.save(Path(RES_DIR, "{}.{}.{}.train.csv".format(exp_name,
                                                                       NGRAM_RANGE_PREFIX,
                                                                       EXP_TYPE)),
                             save_report_fpath=Path(RES_DIR, "{}.{}.{}.train_results.csv".format(exp_name,
                                                                                                 NGRAM_RANGE_PREFIX,
                                                                                                 EXP_TYPE)))
        guess_test_Y = exp_model.predict(test_X)
        test_telemetry = GuessGoldTelemetry(guess_test_Y, test_Y)
        test_telemetry.save(Path(RES_DIR, "{}.{}.{}.test.csv".format(exp_name,
                                                                     NGRAM_RANGE_PREFIX,
                                                                     EXP_TYPE)),
                            save_report_fpath=Path(RES_DIR, "{}.{}.{}.test_results.csv".format(exp_name,
                                                                                               NGRAM_RANGE_PREFIX, EXP_TYPE)))

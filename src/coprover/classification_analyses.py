"""
Wrapper for convenient tools for trialing a classification dataset against
multiple SKLearn methods
"""

from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm
from pathlib import Path
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool

from coprover.results_analysis import GuessGoldTelemetry


# Constituent methods

def inst_svc_linear(rand_state=0, tol=1e-5):
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=rand_state, tol=tol))
    return clf

def inst_svc_poly(rand_state=0, tol=1e-5):
    clf = make_pipeline(StandardScaler(), SVC(kernel='poly', degree=3, random_state=rand_state, tol=tol))
    return clf

def inst_svc_rbf(rand_state=0, tol=1e-5):
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=rand_state, tol=tol))
    return clf

def inst_randforest(n_estimators=10):
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=n_estimators))
    return clf

def inst_knn(n_neighbors=5):
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))
    return clf

CLFs = list({ "svc_linear": inst_svc_linear, "svc_poly": inst_svc_poly, 
         "svc_rbf": inst_svc_rbf, 
         "randforest": inst_randforest, 
         "knn": inst_knn }.items())


def run_suite(X_train, Y_train, X_test, Y_test, 
              save_dir=None, num_workers=len(CLFs),
              verbose=True):
    def run_exp_inner(exp_pair):
        (name, clf_fn) = exp_pair
        clf = clf_fn()
        return run_experiment(X_train, Y_train, X_test, Y_test, clf, name)
    with ThreadPool(num_workers) as p:
        gg_telems = list(tqdm(p.imap_unordered(run_exp_inner, CLFs)))
    if save_dir is not None:
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        for gg_telem in gg_telems:
            gg_telem.save(Path(save_dir, "{}.csv".format(gg_telem.name)))
    gg_telems.sort(key=lambda x: x.name)
    if verbose:
        for gg_telem in gg_telems:
            print("Name:{}\n{}".format(gg_telem.name, gg_telem.class_report(return_dict=False)))
    return gg_telems


def run_experiment(X_train, Y_train, X_test, Y_test, clf, name):
    clf.fit(X_train, Y_train)
    Y_guess = clf.predict(X_test)
    gg_telem = GuessGoldTelemetry(Y_guess, Y_test, name=name)
    return gg_telem



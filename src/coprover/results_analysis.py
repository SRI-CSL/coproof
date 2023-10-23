from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from scipy.stats import t
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""
Combined utilities for recording guess/golds emitted by system and estimating moments of
accuracies via boostrap resampling.
"""

# Setup a telemetry module that will record guess and golds, allowing them to be saved out
GUESS = "guess"
GOLD = "gold"

# Fields for storing results in bootstrapped moments dataframe
EXPNAME = "expname"
MEAN_ACC = "mean_acc"
STD_ACC = "std_acc"
N_COL="N"

class GuessGoldTelemetry:
    """
    Add this telemetry recorder to record guesses performed by
    the system to examine
    """
    def __init__(self, guesses=[], golds=[], target_names=None, name=None):
        assert len(guesses) == len(golds)
        self.guesses = guesses
        self.golds = golds
        self.target_names = target_names
        self.name = name
        
    def __len__(self):
        return len(self.guesses)
    
    def add(self, guess, gold):
        self.guesses.append(guess)
        self.golds.append(gold)
        
    def labels(self):
        """ Returns ordered list of all unique guess/gold values"""
        uniques = set()
        uniques.update(self.guesses)
        uniques.update(self.golds)
        uniques = list(uniques)
        return sorted(uniques)

    def acc(self):
        assert len(self.guesses) == len(self.golds)
        if len(self.guesses) == 0:
            return 0.
        num_right = 0
        for guess, gold in zip(self.guesses, self.golds):
            if guess == gold:
                num_right += 1
        return num_right / len(self.guesses)

    def save(self, fpath, save_report_fpath=None):
        res_df = pd.DataFrame({
            GUESS: self.guesses,
            GOLD: self.golds
        })
        res_df.to_csv(fpath, index=False, header=True)
        if save_report_fpath is not None:
            classification_report = self.class_report()
            class_df = pd.DataFrame(classification_report).transpose()
            class_df.to_csv(save_report_fpath)

    def load(self, fpath, name=None):
        df = pd.read_csv(fpath)
        self.guesses = df[GUESS]
        self.golds = df[GOLD]
        if name is not None:
            self.name = name
        else:
            self.name = Path(fpath).stem

    def confusion_matrix(self, plot=False):
        cm = confusion_matrix(self.golds, self.guesses)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels())
        if plot:
            _, ax = plt.subplots(figsize=(100,100))
            disp.plot(ax=ax)
        return cm

    def class_report(self, return_dict=True):
        return classification_report(self.golds, self.guesses,
                              target_names=self.target_names, output_dict=return_dict)
    
    def filter(self, min_N=None):
        """
        Filters this, returning only guess/gold entries that are above the cutoff count
        """
        # Generate histogram of all guess/gold
        # pair_freqs = Counter(zip(self.guesses, self.golds))
        freqs = Counter(self.golds)
        updated_guesses = []
        updated_golds = []
        for guess, gold in zip(self.guesses, self.golds):
            freq = min(freqs[guess], freqs[gold])
            if freq >= min_N:
                updated_guesses.append(guess)
                updated_golds.append(gold)
        assert len(updated_guesses) == len(updated_golds)
        assert len(updated_guesses) <= len(self.guesses)
        return GuessGoldTelemetry(updated_guesses, updated_golds, 
                                  target_names=self.target_names,
                                  name="{} filtered N={}".format(self.name, min_N))


def read_gg_from_csv(csv_fpath):
    """
    Given a path to a GuessGoldTelemetry csv file, reconstitutes it.
    """
    gg = GuessGoldTelemetry()
    gg.load(csv_fpath)
    return gg

#
# Boostrap moment estimator code
#
def compute_acc(results_df):
    num_correct = np.sum(results_df[GUESS] == results_df[GOLD])
    acc = num_correct / len(results_df)
    return acc

def bootstrap_acc_stats(fpath, num_samples=1000, expname=None):
    """
    Loads in the CSV or referenced dataframe following the given convention
    and runs num_samples bootstrap resamplings to estimate moments.
    """
    if isinstance(fpath, pd.DataFrame):
        results_df = fpath
    else:
        results_df = pd.read_csv(fpath)    
        expname = fpath.stem
    accs = []
    for _ in range(num_samples):
        sampled_df = results_df.sample(frac=1, replace=True)
        acc = compute_acc(sampled_df)
        accs.append(acc)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    total = len(accs)
    return mean_acc, std_acc, total

def _is_test_fn(csv_fpath):
    """
    Convenience routine, checks if the referenced results CSV has 'test' in the name.
    """
    return "test" in csv_fpath.stem

def bootstrap_csvs_in_dir(fpath, csv_fpath_check=_is_test_fn, **kwargs):
    """
    Examines all CSVs in fpath dir and returns an accumulated set of moments 
    for each CSV result file.  Note: presumes all CSVs are results files in the
    given format.
    """
    csv_files = Path(fpath).glob("*.csv")
    accum_res = []
    for csv_fpath in tqdm(sorted(csv_files)):
        if csv_fpath_check(csv_fpath):
            mean_acc, std_acc, total = bootstrap_acc_stats(csv_fpath, **kwargs)
            accum_res.append({
                EXPNAME: csv_fpath.stem,
                MEAN_ACC: mean_acc,
                STD_ACC: std_acc,
                N_COL: total
            })
    all_results = pd.DataFrame(accum_res)
    return all_results


def two_t(mu1, s1, n1, mu2, s2, n2):
    # Given the means (mu), standard deviation (s), and sample size (n) for two distributions,
    # Determines p-value of the measure.  If within the critical values, then are not significant
    pv = ((n1 - 1) * s1 ** 2) + ((n2 - 1) * s2 ** 2)
    pv /= (n1 + n2 - 2)
    ps = np.sqrt(pv)
    return (mu1 - mu2) / (ps * np.sqrt(1/n1 + 1/n2))

def get_crit_vals(alpha, N):
    crit_val = t.ppf(1 - alpha / 2, N)
    crit_val2 = t.ppf(alpha / 2, N)
    print("Critical value = {:.3f}/{:.3f}, two tailed alpha={}, N={}".format(crit_val, crit_val2, alpha, N))
    return crit_val, crit_val2


def sig_test(df, exp1, exp2, alpha=0.05):
    """ Convenience routine for comparing t-value and critical value of two
    experiments in a results dataframe computed by bootstrap_csvs_in_dir 
    :param: exp1 : Experiment name of the first experiment
    :param: exp2 : Experiment name for the second to compare against."""
    if exp1 not in set(df[EXPNAME]):
        raise Exception("No experiment name found for exp1={}".format(exp1))
    if exp2 not in set(df[EXPNAME]):
        raise Exception("No experiment name found for exp2={}".format(exp2))        
    row1 = df[df[EXPNAME] == exp1].iloc[0]
    row2 = df[df[EXPNAME] == exp2].iloc[0]
    p_value = two_t(row1[MEAN_ACC], row1[STD_ACC], row1[N_COL], 
                    row2[MEAN_ACC], row2[STD_ACC], row2[N_COL])
    N = min(row1[N_COL], row2[N_COL])
    crit_val_bounds = get_crit_vals(alpha, N)
    return p_value, crit_val_bounds
    

NAME = "name"
MACRO_AVG = "macro avg"
MICRO_AVG = "weighted avg"
F1 = "f1-score"
MACRO_F1 = "macro_f1"
MICRO_F1 = "micro_f1"
ACC = "accuracy"

def summarize(gg_telems):
    """ Given a set of GG telems, summarizes the results"""
    summary = []
    for gg_telem in gg_telems:
        assert(isinstance(gg_telem, GuessGoldTelemetry))
        report = gg_telem.class_report(return_dict=True)
        name = gg_telem.name
        macro_f1 = report[MACRO_AVG][F1]
        weighted_f1 = report[MICRO_AVG][F1]
        summary.append({
            NAME: name,
            MACRO_F1: macro_f1,
            MICRO_F1: weighted_f1,
            ACC: report[ACC]
        })
    return pd.DataFrame(summary)

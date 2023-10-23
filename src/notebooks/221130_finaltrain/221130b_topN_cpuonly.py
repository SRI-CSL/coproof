"""
Generates the top-N plot
"""
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from coprover.cmdpred.cmdpred_data_setup import *
from coprover.cmdpred import CmdPredictor

EXP_TYPE = FULL
RES_DIR = Path("results_cpu")
RES_DIR.mkdir(exist_ok=True, parents=True)

def acc_at_N(df, model, N=5):
    """ Gets accuracy @ N"""
    tp = 0
    model.model.eval()
    tqdm_iter = tqdm(df.iterrows(), total=len(df))
    num_seen = 0
    for idx, row in tqdm_iter:
        gold = row[TGT_TXT]
        guesses = [normalize_cmd(pref) for pref in model.predict(row[SRC_TXT], num_return_sequences=N, num_beams=2 * N)]
        if gold in guesses:
            tp += 1
        num_seen += 1
        tqdm_iter.set_description("{}/{}, {:.3f}".format(tp, num_seen, tp/num_seen))
    return tp / len(df)

train_df, test_df = get_splits(exp_type=EXP_TYPE)
cmdpredictor = CmdPredictor(model_fpath="outputs/cmd_pred1_N3/simplet5-epoch-9-train-loss-0.4682-val-loss-0.4904", use_gpu=False, use_device="cuda:2")

accs = []
for n in range(1, 11):
    curr_acc = acc_at_N(test_df.iloc[0:1000], cmdpredictor.model, N=n)
    print("N={}: Acc={:.5f}".format(n, curr_acc))
    accs.append((n, curr_acc))

N = "N"
ACC = "Acc"

results_df = pd.DataFrame([ { N: x[0], ACC: x[1]} for x in accs ])
results_df.to_csv(Path(RES_DIR, "topN_T5_{}.csv".format(EXP_TYPE)))

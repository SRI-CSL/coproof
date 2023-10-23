"""
Generates the top-N plot
"""
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from coprover import RSC_ROOT
from coprover.cmdpred.cmdpred_data_setup import *
from coprover.cmdpred import CmdPredictor

EXP_TYPE = CMDHISTONLY

if EXP_TYPE == FULL:
    EMODEL_FPATH = Path(RSC_ROOT, "pvs_cmd_pred/pvslib/pvslib_cmdpred_t5_full/simplet5-epoch-9-train-loss-0.4691-val-loss-0.4577")
elif EXP_TYPE == SEQONLY:    
    MODEL_FPATH = Path(RSC_ROOT, "pvs_cmd_pred/pvslib/pvslib_cmdpred_t5_sequentonly/simplet5-epoch-9-train-loss-0.7113-val-loss-0.6314")
elif EXP_TYPE == CMDHISTONLY:
    MODEL_FPATH = Path(RSC_ROOT, "pvs_cmd_pred/pvslib/pvslib_cmdpred_t5_cmdhistonly/simplet5-epoch-9-train-loss-0.6419-val-loss-0.5901")

RES_DIR = Path("results")
RES_DIR.mkdir(exist_ok=True, parents=True)

def acc_at_N(df, model, N=5):
    """ Gets accuracy @ N"""
    tp = 0
    model.model.eval()
    tqdm_iter = tqdm(df.iterrows(), total=len(df))
    num_seen = 0
    for idx, row in tqdm_iter:
        gold = row[TGT_TXT]
        num_toks = len(row[SRC_TXT].split())
        guesses = [normalize_cmd(pref) for pref in model.predict(row[SRC_TXT], num_return_sequences=N, num_beams=2 * N)]
        if gold in guesses:
            tp += 1
        num_seen += 1
        tqdm_iter.set_description("{}/{}, {:.3f}".format(tp, num_seen, tp/num_seen))
    return tp / len(df)

train_df, test_df = get_splits(exp_type=EXP_TYPE)
cmdpredictor = CmdPredictor(model_fpath=MODEL_FPATH, use_gpu=True, use_device="cuda:3", max_src_tok_len=1000)

N = "N"
ACC = "Acc"
accs_at_N = []    
for n in range(1, 11):
    curr_acc = acc_at_N(test_df, cmdpredictor.model, N=n)
    print("N={}: Acc={:.5f}".format(n, curr_acc))
    accs_at_N.append((n, curr_acc))
    results_df = pd.DataFrame([ { N: x[0], ACC: x[1]} for x in accs_at_N ])
    results_df.to_csv(Path(RES_DIR, "topN={}_T5_{}.csv".format(N, EXP_TYPE)))

"""
Generates the guesses and golds for computing suff stats
"""
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from coprover import RSC_ROOT
from coprover.results_analysis import *
from coprover.cmdpred.cmdpred_data_setup import *
from coprover.cmdpred import CmdPredictor


for EXP_TYPE in (FULL, SEQONLY, CMDHISTONLY):
    print("Running exptype={}".format(EXP_TYPE))
    if EXP_TYPE == FULL:
        MODEL_FPATH = Path(RSC_ROOT, "pvs_cmd_pred/pvslib/models/pvslib_cmdpred_t5_full/simplet5-epoch-9-train-loss-0.4691-val-loss-0.4577")
    elif EXP_TYPE == SEQONLY:    
        MODEL_FPATH = Path(RSC_ROOT, "pvs_cmd_pred/pvslib/models/pvslib_cmdpred_t5_sequentonly/simplet5-epoch-9-train-loss-0.7113-val-loss-0.6314")
    elif EXP_TYPE == CMDHISTONLY:
        MODEL_FPATH = Path(RSC_ROOT, "pvs_cmd_pred/pvslib/models/pvslib_cmdpred_t5_cmdhistonly/simplet5-epoch-9-train-loss-0.6419-val-loss-0.5901")

    RES_DIR = Path("results")
    RES_DIR.mkdir(exist_ok=True, parents=True)

    train_df, test_df = get_splits(exp_type=EXP_TYPE)
    cmdpredictor = CmdPredictor(model_fpath=MODEL_FPATH, use_gpu=True, use_device="cuda:0", max_src_tok_len=1000)

    telemetry_train = GuessGoldTelemetry()
    cmdpredictor.model.model.eval()
    tqdm_iter = tqdm(train_df.iterrows(), total=len(train_df))
    num_seen = 0
    for idx, row in tqdm_iter:
        gold = row[TGT_TXT]
        num_toks = len(row[SRC_TXT].split())
        guess = cmdpredictor.model.predict(row[SRC_TXT])[0]
        telemetry_train.add(guess, gold)
    telemetry_train.save(Path(RES_DIR, "t5.train.{}.csv".format(EXP_TYPE)))

    telemetry_test = GuessGoldTelemetry()
    cmdpredictor.model.model.eval()
    tqdm_iter = tqdm(test_df.iterrows(), total=len(test_df))
    num_seen = 0
    for idx, row in tqdm_iter:
        gold = row[TGT_TXT]
        num_toks = len(row[SRC_TXT].split())
        guess = cmdpredictor.model.predict(row[SRC_TXT])[0]
        telemetry_test.add(guess, gold)
    telemetry_test.save(Path(RES_DIR, "t5.test.{}.csv".format(EXP_TYPE)))


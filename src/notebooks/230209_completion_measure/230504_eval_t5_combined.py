#!/usr/bin/env python
# coding: utf-8

# This evaluates the last command prediction model trained on both V1 and V2 of the data, around different combinations.
# 
# 
# 


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report

from exp_setup import *
from coprover.results_analysis import GuessGoldTelemetry
from coprover.training.simplet5 import SimpleT5

MODELS_DIR = Path("outputs")

V2_MODELS_DIR = Path("outputs", "t5", "v2")
RESULTS_DIR = Path("results", "t5")

DEVICE = "cuda:1"

# Model listing, tuple (strip_cmdhistory, cmds_only, path)
MODELS = {
    "t5_full_v2": (False, False, Path(V2_MODELS_DIR, "laststep_pred_v2", "best_model")),
    "t5_cmdsonly_v2": (False, True, Path(V2_MODELS_DIR, "laststep_pred_cmdsonly_v2", "best_model")),
    "t5_nocmds_v2": (True, False, Path(V2_MODELS_DIR, "laststep_pred_nocmds_v2", "best_model")),
    
    "t5_full_v1": (False, False, Path(MODELS_DIR, "laststep_pred_v1", "best_model")),
    "t5_cmdsonly_v1": (False, True, Path(MODELS_DIR, "laststep_pred_cmdsonly_v1", "best_model")),
    "t5_nocmds_v1": (True, False, Path(MODELS_DIR, "laststep_pred_nocmds_v1", "best_model"))
}

def norm_label(label):
    if label.startswith(NEG):
        return NEG
    if label.startswith(POS):
        return POS
    print(f"Alert, nonstandard label={label}, using default majority train NEG")
    return NEG

def run_experiment(exp_name, strip_cmdhistory, cmds_only, model_fpath):
    print(f"Running experiment: {exp_name}, model_fpath={model_fpath}, strip_cmdhistory={strip_cmdhistory}, cmds_only={cmds_only}")
    model = SimpleT5(source_max_token_len=SRC_MAX_TOKLEN, target_max_token_len=TGT_MAX_TOKLEN)
    model.load_model(model_fpath, use_gpu=True, use_device=DEVICE)
    train_df, test_df = setup_laststep_pred_data(strip_cmdhistory=strip_cmdhistory, cmds_only=cmds_only)
    print(f"Len train={len(train_df)}, test={len(test_df)}")
    print(f"Train, # pos={np.sum(train_df.target_text == POS)}, neg={np.sum(train_df.target_text == NEG)}")
    print(f"Test, # pos={np.sum(test_df.target_text == POS)}, neg={np.sum(test_df.target_text == NEG)}")
    
    # Eval Test
    test_Y_guess = []
    test_Y_gold = test_df.target_text.array
    for src_txt in tqdm(test_df.source_text.array):
        test_Y_guess.append(norm_label(model.predict(src_txt)[0]))
    test_telem = GuessGoldTelemetry(guesses=test_Y_guess, golds=test_Y_gold, target_names=[NEG, POS], 
                                    name="{exp_name} Test")
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    test_telem.save(Path(RESULTS_DIR, f"{exp_name}.csv"))
    print("Test Result")
    print(test_telem.class_report(return_dict=False))
    # Subsample train
    idx = 1000
    train_Y_guess = []
    train_Y_gold = train_df.target_text.array[0:idx]
    for src_txt in tqdm(train_df.source_text.array[0:idx]):
        train_Y_guess.append(norm_label(model.predict(src_txt)[0]))
    train_telem = GuessGoldTelemetry(guesses=train_Y_guess, golds=train_Y_gold, target_names=[NEG, POS], 
                                     name=f"{exp_name} Train(0:{idx})")
    print("Train Result")
    print(train_telem.class_report(return_dict=False))
    return test_telem, train_telem
    


for exp_name, exp_tuple in MODELS.items():
    strip_cmdhistory, cmds_only, model_fpath = exp_tuple
    test_telem, train_telem = run_experiment(exp_name, strip_cmdhistory, cmds_only, model_fpath)


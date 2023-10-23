import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report

from exp_setup import *
from coprover.training.simplet5 import SimpleT5
from coprover.results_analysis import GuessGoldTelemetry

DEVICE="cuda:1"

T5_MODEL_FPATH = Path("outputs", "laststep_pred_v2", "best_model")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True, parents=True)

model = SimpleT5(source_max_token_len=SRC_MAX_TOKLEN, target_max_token_len=TGT_MAX_TOKLEN)
model.load_model(T5_MODEL_FPATH, use_gpu=True, use_device=DEVICE)

train_df, test_df = setup_laststep_pred_data()

Y_guess = []
Y_gold = test_df.target_text.array
for src_txt in tqdm(test_df.source_text.array):
    Y_guess.append(model.predict(src_txt)[0])

print(classification_report(Y_gold, Y_guess))
# print(classification_report(Y_gold, Y_guess, target_names=['neg', 'pos']))

gg_telem = GuessGoldTelemetry(Y_guess, Y_gold)
gg_telem.save(Path(results_dir, "t5_combined_pred.csv"))

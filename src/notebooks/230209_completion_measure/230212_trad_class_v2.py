"""
Run traditional class analyses on V2 data
"""
import numpy as np
from pathlib import Path
from exp_setup import *
from coprover.classification_analyses import run_suite
from coprover.results_analysis import summarize

resdir = Path("results", "data_v2")
resdir.mkdir(exist_ok=True, parents=True)

CSV_FILE = "laststep_pred.v2.csv"

tfidf_X_train, tfidf_Y_train, tfidf_X_test, tfidf_Y_test = setup_state_dict_data(csv_file=CSV_FILE, use_tfidf=True)
count_X_train, count_Y_train, count_X_test, count_Y_test = setup_state_dict_data(csv_file=CSV_FILE, use_tfidf=False)
cmd_X_train, cmd_Y_train, cmd_X_test, cmd_Y_test = setup_cmd_dict_data()

combined_X_train = np.concatenate([tfidf_X_train, cmd_X_train], axis=1)
combined_X_test = np.concatenate([tfidf_X_test, cmd_X_test], axis=1)

combined_gg_telems = run_suite(combined_X_train, tfidf_Y_train, combined_X_test, tfidf_Y_test,
                            save_dir=Path(resdir, "combined" ) )
combined_summary_df = summarize(combined_gg_telems)
combined_summary_df.to_csv(Path(resdir, "combined_summary.csv"))


tfidf_gg_telems = run_suite(tfidf_X_train, tfidf_Y_train, tfidf_X_test, tfidf_Y_test,
                            save_dir=Path(resdir, "state_tfidf" ) )
tfidf_summary_df = summarize(tfidf_gg_telems)
tfidf_summary_df.to_csv(Path(resdir, "tfidf_summary.csv"))

count_gg_telems = run_suite(count_X_train, count_Y_train, count_X_test, count_Y_test,
                            save_dir=Path(resdir, "state_count" ) )
count_summary_df = summarize(count_gg_telems)
count_summary_df.to_csv(Path(resdir, "count_summary.csv"))


combined2_X_train = np.concatenate([count_X_train, cmd_X_train], axis=1)
combined2_X_test = np.concatenate([count_X_test, cmd_X_test], axis=1)
combined2_gg_telems = run_suite(combined2_X_train, tfidf_Y_train, combined2_X_test, tfidf_Y_test,
                            save_dir=Path(resdir, "combined2" ) )
combined2_summary_df = summarize(combined2_gg_telems)
combined2_summary_df.to_csv(Path(resdir, "combined2_summary.csv"))



tfidf_gg_telems = run_suite(tfidf_X_train, tfidf_Y_train, tfidf_X_test, tfidf_Y_test,
                            save_dir=Path(resdir, "state_tfidf" ) )
tfidf_summary_df = summarize(tfidf_gg_telems)
tfidf_summary_df.to_csv(Path(resdir, "tfidf_summary.csv"))

cmd_gg_telems = run_suite(cmd_X_train, cmd_Y_train, cmd_X_test, cmd_Y_test, 
                          save_dir=Path(resdir, "cmd_clf"), verbose=True)
res_summary_df = summarize(cmd_gg_telems)
res_summary_df.to_csv(Path(resdir, "cmd_std_summary.csv"))


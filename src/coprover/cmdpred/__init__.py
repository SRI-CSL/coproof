"""
Command Prediction utilities

Based off of notebooks from 220325_t5
"""
from pathlib import Path
import json

from coprover import RSC_ROOT
from coprover.training.simplet5 import SimpleT5
from coprover.feats.commands import normalize
from coprover.feats.featurize_cmdpred import format_state

ANT_TOK = "<ANT>"
CONS_TOK = "<CONS>"
HIDDEN_TOK = "<HID>"
NOOP_CMD = "NOOP"


CMD1_PREFIX = "command1: "
PAD_TOK = " <pad> "

DEFAULT_MODEL_FPATH = Path(RSC_ROOT, "pvs_cmd_pred", "models", "cmd_pred1_hist3", "curr_best")


def format_input(state, cmd_history):
    if isinstance(state, dict):
        state = format_state(state)
    if isinstance(cmd_history, str):
        # If given as a strict string, convert to list
        cmd_history = [tok.strip() for tok in cmd_history.split(",")]
    return CMD1_PREFIX + ",".join(cmd_history) + PAD_TOK + state


class CmdPredictor:
    def __init__(self, model_fpath=DEFAULT_MODEL_FPATH, max_src_tok_len=1000, 
                 use_gpu=True, use_device="cuda:0", cmd_history_N=3,
                 **kwargs):
        self.model = SimpleT5(source_max_token_len=max_src_tok_len,
                         target_max_token_len=10, **kwargs)
        self.model.load_model(model_fpath, use_gpu=use_gpu,
                              use_device=use_device)
        self.model_fpath = model_fpath
        self.max_src_tok_len = max_src_tok_len
        self.cmd_history_N = cmd_history_N

    def predict(self, state_json, cmd_history=[], N=5):
        if isinstance(state_json, str):
            state_json = json.loads(state_json)
        if cmd_history is None or len(cmd_history) == 0:
            cmd_history = [NOOP_CMD for _ in range(self.cmd_history_N)]  # Default starting is N=3
        state = format_state(state_json)
        input_str = CMD1_PREFIX + ",".join(cmd_history) + PAD_TOK + state
        raw_guesses = self.model.predict(input_str,
                           num_return_sequences=N, num_beams = 2 * N)
        guesses = [normalize(guess) for guess in raw_guesses]
        return guesses

    def __str__(self):
        return "CmdPred, command history={}, model_fpath={}, max src tok len={}".format(self.cmd_history_N, self.model_fpath,
                                                                                       self.max_src_tok_len)

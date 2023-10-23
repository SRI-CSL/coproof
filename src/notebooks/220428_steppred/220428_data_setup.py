"""
Data setup for the step/phase prediction experiment

This is derived from 220325_t5/220325_data_setup.py

Reads in the JSON formatted data and constructs the TSV format
used for training seq2seq models.

This uses the same featurization, but attempts to predict the
progress along the proof.

"""

import re
import json
from tqdm import tqdm
from pathlib import Path
from queue import Queue

from coprover import PROJ_ROOT
from coprover.feats.parsing.prf_json import convert_dict2tags, Tag

DATA_ROOT = Path(PROJ_ROOT, "data", "pvs", "prelude")


def print_tags(sform):
    ret = ""
    if isinstance(sform, Tag):
        ret += " {}".format(sform.tag)
        for k, v in sform.elts.items():
            ret += print_tags(v)
    elif isinstance(sform, list):
        for v in sform:
            ret += " {}".format(print_tags(v))
    ret.replace("  ", " ") # Hack to remove double spaces, need more clever way
    return ret


ANT_TOK = "<ANT>"
CONS_TOK = "<CONS>"
HIDDEN_TOK = "<HID>"


def format_state(state):
    antecedents = state.elts['current-goal'].elts['antecedents']
    consequents = state.elts['current-goal'].elts['consequents']
    hidden = state.elts['current-goal'].elts['hidden']
    ret = [ANT_TOK,
           print_tags(antecedents),
           CONS_TOK,
           print_tags(consequents),
           HIDDEN_TOK,
           print_tags(hidden)]
    ret = " ".join(ret)
    return re.sub("\\s+", " ", ret)


def _is_valid_state(state, command):
    """
    Checks if this is a state and/or command we do not wish to target
    :param state:
    :param command:
    :return:
    """
    if command == "assert":
        return False
    elif command.endswith("-tcc"):
        return False
    elif command == "tcc":
        return False
    return True


NOOP_CMD = "NOOP"

def process_json(fpath, tel_f, N=3):
    with open(fpath, 'r') as f:
        prf = json.load(f)
    root = convert_dict2tags(prf)
    if 'proof' in root.elts:
        proof_steps = root.elts['proof']
        if proof_steps is None:
            print("Degenerate proof={}".format(fpath))
            return None
        else:
            # Process this proof's steps
            cmd_buffer = Queue()
            for _ in range(N):
                cmd_buffer.put(NOOP_CMD)
            sa_pairs = [(proof_steps[i], proof_steps[i+1]) for i in range(0, len(proof_steps), 2)]
            total_steps = len(sa_pairs)
            stepnum = 0
            for state, input in sa_pairs:
                command = input[1]
                stepnum += 1
                #if _is_valid_state(state, command):  # Removed filtering over 'assert'
                tel_f.write("{}\t".format(format_state(state)))
                tel_f.write("{}".format(input[1]))
                if N > 0:
                    # If we are recording the history, then emit current window and
                    # then push the current command on after
                    cmd_window = cmd_buffer.queue
                    assert len(cmd_window) == N
                    tel_f.write("\t{}".format(",".join(cmd_window)))
                    cmd_buffer.put(command)
                    cmd_buffer.get()
                tel_f.write("\t{:.3f}".format(stepnum / total_steps))
                tel_f.write("\n")
            return len(sa_pairs)


json_files = list(DATA_ROOT.rglob("*.json"))

tqdm_iter = tqdm(json_files, total=len(json_files))
CMD_N=3
tel_f = open("tags_proofstep_cmdN={}.tsv".format(CMD_N), 'w')
num_steps = 0
num_valid_proofs = 0
num_empty = 0
for json_fpath in tqdm_iter:
    tqdm_iter.set_description("{}".format(json_fpath))
    proof_steps = process_json(json_fpath, tel_f, N=CMD_N)
    if proof_steps is None:
        num_empty += 1
    else:
        num_steps += proof_steps
        num_valid_proofs += 1
tel_f.close()

print("Total steps processed = {}, num valid proofs={}".format(num_steps, num_valid_proofs))
print("Total empty proof dumps={}".format(num_empty))
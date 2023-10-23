"""
Command predicate prediction featurization.

Derived from 220325_t5/220325_data_setup.py

Running this as the main module generates TSVs for the PVS prelude and
full PVSlib into the CoProver/results directory.
"""
import re
import json
from queue import Queue
from tqdm import tqdm
from pathlib import Path

from coprover import PROJ_ROOT
from coprover.feats.parsing.prf_json import convert_dict2tags, Tag

ANT_TOK = "<ANT>"
CONS_TOK = "<CONS>"
HIDDEN_TOK = "<HID>"
NOOP_CMD = "NOOP"

# Tuple containing the special tokens for command pred experiment formatting
VOCAB_SPEC_TOKS = (ANT_TOK, CONS_TOK, HIDDEN_TOK, NOOP_CMD)

DEPTH = "depth"

def print_tags(sform):
    ret = ""
    if isinstance(sform, Tag):
        ret += " {}".format(sform.tag)
        if sform.depth is not None:
            ret += " DEPTH {}".format(sform.depth)
        for k, v in sform.elts.items():
            ret += print_tags(v)
    elif isinstance(sform, list):
        for v in sform:
            ret += " {}".format(print_tags(v))
    ret.replace("  ", " ") # Hack to remove double spaces, need more clever way
    return ret


def format_state(state):
    if isinstance(state, dict):
        state = convert_dict2tags(state)
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
    if state is None:
        return False
    if command == "assert":
        return False
    elif command.endswith("-tcc"):
        return False
    elif command == "tcc":
        return False
    return True


class ProofStateAction:
    def __init__(self, state, command, uri, cmd_buffer=None, depth=None):
        self.raw_state = state
        self.state, self.command = format_state(state), command
        self.cmd_buffer = cmd_buffer
        self.uri = uri
        self.depth = depth

    def to_tsv(self):
        if self.cmd_buffer is None:
            return "\t".join((self.state, self.command, self.uri, str(self.depth)))
        else:
            cmdhist = ",".join(self.cmd_buffer)
            return "\t".join((self.state, self.command, cmdhist, self.uri, str(self.depth)))
        
    def to_pretty(self):
        if self.cmd_buffer is None:
            return "\t".join((self.raw_state.pprint(), self.command, self.uri, str(self.depth)))
        else:
            cmdhist = ",".join(self.cmd_buffer)
            return "\t".join((self.raw_state.pprint(), self.command, cmdhist, self.uri, str(self.depth)))


def get_command(input):
    """
    Returns the command label from the given input.  This handles the case where the
    input is a list, or is a JSON structure.
    :param input:
    :return:
    """
    if isinstance(input, Tag):
        return input.elts['rule']
    elif isinstance(input, list):
        return input[1]
    raise Exception("Unknown type for input={}, input={}".format(type(input), input))

import pdb
def process_json(prf_json, uri_base, N=3):
    root = convert_dict2tags(prf_json)
    if 'proof' in root.elts:
        proof_steps = root.elts['proof']
        if proof_steps is None:
            return []
        else:
            # Process this proof's steps
            cmd_buffer = Queue()
            for _ in range(N):
                cmd_buffer.put(NOOP_CMD)
            raw_sa_pairs = [(proof_steps[i], proof_steps[i+1]) for i in range(0, len(proof_steps), 2)]
            episode = []
            for idx, (state, input_str) in enumerate(raw_sa_pairs):
                command = get_command(input_str)
                if _is_valid_state(state, command):
                    uri = "{}#{}".format(uri_base, idx)
                    episode.append(ProofStateAction(state, command, uri,
                                                    cmd_buffer=list(cmd_buffer.queue),
                                                    depth=state.depth))
                    if N > 0:
                        # If we are recording the history, then emit current window and
                        # then push the current command on after
                        cmd_window = cmd_buffer.queue
                        assert len(cmd_window) == N
                        cmd_buffer.put(command)
                        cmd_buffer.get()
            return episode
    return []


def process_file(fpath, N=3):
    uri_base = "{}/{}".format(Path(fpath).parent.name, Path(fpath).stem)
    with open(fpath, 'r') as f:
        prf_stateaction_pairs = process_json(json.load(f), uri_base, N=N)
        if len(prf_stateaction_pairs) == 0:
            print("Degenerate proof trace={}".format(fpath))
        return prf_stateaction_pairs


def process_json_files(json_files, output_fpath, N=3, pretty_state=False):
    Path(output_fpath).parent.mkdir(parents=True, exist_ok=True)
    with open(output_fpath, 'w') as f:
        tqdm_iter = tqdm(json_files, total=len(json_files))
        for json_fpath in tqdm_iter:
            tqdm_iter.set_description("{}".format(json_fpath))
            episode = process_file(json_fpath, N=N)
            if len(episode) > 0:
                for stateaction in episode:
                    if pretty_state:
                        f.write(stateaction.to_pretty())
                    else:
                        f.write(stateaction.to_tsv())
                    f.write("\n")


def process_prelude(results_fpath=Path("results/cmdpred_N3.prelude.tsv")):
    DATA_ROOT = Path(PROJ_ROOT, "data", "pvs", "prelude")
    json_files = list(DATA_ROOT.rglob("*.json"))
    process_json_files(json_files, results_fpath)


def process_pvslib(results_fpath=Path("results/cmdpred_N3.pvslib.tsv")):
    DATA_ROOT = Path(PROJ_ROOT, "data", "pvs", "pvslib")
    json_files = list(DATA_ROOT.rglob("*.json"))
    process_json_files(json_files, results_fpath)


                                
def process_prelude_pretty(results_fpath=Path("results/cmdpred_N3.prelude.pretty.tsv")):
    DATA_ROOT = Path(PROJ_ROOT, "data", "pvs", "prelude")
    json_files = list(DATA_ROOT.rglob("*.json"))
    process_json_files(json_files, results_fpath, pretty_state=True)


def process_pvslib_pretty(results_fpath=Path("results/cmdpred_N3.pvslib.pretty.tsv")):
    DATA_ROOT = Path(PROJ_ROOT, "data", "pvs", "pvslib")
    json_files = list(DATA_ROOT.rglob("*.json"))
    process_json_files(json_files, results_fpath, pretty_state=True)
                                
if __name__ == "__main__":
    process_prelude()
    process_prelude_pretty()
    process_pvslib()
    process_pvslib_pretty()                                

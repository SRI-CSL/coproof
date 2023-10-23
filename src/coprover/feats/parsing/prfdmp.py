"""
Handles processing directly from S-Expression format.

Deprecated in favor of the JSON based export, but retained for
ability to work directly with s-expressions.
"""


from enum import Enum
import dataclasses
from dataclasses import dataclass
from coprover.feats.parsing.sexp import parse

class ParseState(Enum):
    STATE = 0
    ACTION = 1
    TYPEHASH = 2

@dataclass
class SAPair:
    state:list = dataclasses.field(default_factory=list)
    action:list = dataclasses.field(default_factory=list)

    def command(self):
        return self.action[0]


PRFSTATE_PREFIX = "(proofstate"
TYPEHASH_PREFIX = "Typehash for"
DELIM = ":::"


def process(prf_file):
    """ Reads in the prf_file filepath and generates a state-action pair"""
    parse_state = ParseState.STATE
    with open(prf_file, 'r') as f:
        sa_raw_pairs = []
        curr_state = []
        curr_action = []
        for line in f:
            if line.startswith("Proof dump"):
                continue
            elif parse_state == ParseState.TYPEHASH:
                continue
            elif parse_state == ParseState.ACTION:
                if len(line.strip()) == 0 or line.startswith(TYPEHASH_PREFIX) or \
                        line.startswith(PRFSTATE_PREFIX):
                    parse_state = ParseState.STATE
                    if len(curr_state) > 0:
                        sa_raw_pairs.append((curr_state, curr_action))
                        curr_state = []
                        curr_action = []
                    if line.startswith(TYPEHASH_PREFIX):
                        parse_state = ParseState.TYPEHASH
                    elif line.startswith(PRFSTATE_PREFIX):
                        curr_state.append(line)
                else:
                    curr_action.append(line)
            elif parse_state == ParseState.STATE:
                if line.startswith(PRFSTATE_PREFIX):
                    parse_state = ParseState.STATE
                    if len(curr_state) > 0:
                        sa_raw_pairs.append((curr_state, curr_action))
                        curr_state = []
                        curr_action = []
                    curr_state.append(line)
                elif line.startswith(TYPEHASH_PREFIX):
                    parse_state = ParseState.TYPEHASH
                    if len(curr_state) > 0:
                        sa_raw_pairs.append((curr_state, curr_action))
                        curr_state = [line]
                        curr_action = []
                else:
                    line_t = line.split(DELIM)
                    if len(line_t) == 2:
                        parse_state = ParseState.ACTION
                        curr_action.append(line_t[1])
                    curr_state.append(line_t[0])
    sa_pairs = []
    for sa_raw_pair in sa_raw_pairs:
        state_str = " ".join(sa_raw_pair[0])
        action_str = " ".join(sa_raw_pair[1])
        prf_state = parse(state_str)
        action = parse(action_str, is_cmd=True)
        sa_pairs.append(SAPair(prf_state, action))
    return sa_pairs
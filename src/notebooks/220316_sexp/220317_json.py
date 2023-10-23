import json
from pathlib import Path
from collections import OrderedDict
from pprint import pprint

from coprover import PROJ_ROOT
from coprover.feats.parsing.prfdmp import process
DATA_ROOT = Path(PROJ_ROOT, "data", "pvs", "dev")
fpath = Path(DATA_ROOT, "real_props-axiom_of_archimedes.prfdmp")

with open(fpath, 'r') as f:
    prf = json.load(f)

# Attempts at simplifying the JSON
class Tag:
    def __init__(self, tag, label=None):
        self.tag = tag
        self.label = label
        self.elts = OrderedDict()

    def pprint(self):
        elts_str = ""
        for k, v in self.elts.items():
            if isinstance(v, Tag) or isinstance(v, Entries):
                v_str = v.pprint()
            elif isinstance(v, list):
                list_str = ""
                for x in v:
                    list_str += " {}".format(str(x))
                v_str="({})".format(list_str)
            else:
                v_str = str(v)
            elts_str += " :{} {} ".format(k, v_str)
        return "({} {} {})".format(self.tag, self.label,
                                   elts_str)

    def __str__(self):
        return self.pprint()


class Entries(Tag):
    def __init__(self):
        super(Entries, self).__init__(None, None)

    def pprint(self):
        elts_str = ""
        for k, v in self.elts.items():
            elts_str += " :{} {} ".format(k, str(v))
        return elts_str

    def __str__(self):
        return self.pprint()

TAG = 'tag'
LABEL = 'label'
def _is_resv(key):
    return key in set([TAG, LABEL])

def process(exp):
    if isinstance(exp, list):
        return [process(x) for x in exp]
    elif isinstance(exp, str) or isinstance(exp, int) or exp is None:
        return exp
    assert isinstance(exp, dict)
    if 'tag' in exp:
        tag = Tag(exp[TAG], exp.get(LABEL, None))
        for k, v in exp.items():
            if not(_is_resv(k)):
                # Check to see if the value is an un-tagged type.  If so,
                tag.elts[k] = process(v)
        return tag
    else:
        ret = Entries()
        for k, v in exp.items():
            ret.elts[k] = process(v)
        return ret

root = process(prf)

# Next todo
# - Convert apply into actual function calls, e.g. (< x i)
proof_steps = root.elts['proof']
sa_pairs = [(proof_steps[i], proof_steps[i+1]) for i in range(0, len(proof_steps), 2)]

state, input = sa_pairs[0]

def print_tags(sform):
    ret = ""
    if isinstance(sform, Tag):
        ret += " {}".format(sform.tag)
        for k, v in sform.elts.items():
            ret += print_tags(v)
    elif isinstance(sform, list):
        for v in sform:
            ret += " {}".format(print_tags(v))
    return ret

def print_state(state):
    antecedents = state.elts['current-goal'].elts['antecedents']
    consequents = state.elts['current-goal'].elts['consequents']
    hidden = state.elts['current-goal'].elts['hidden']
    return "{}\n|----\n{}\nHidden:\n{}".format(print_tags(antecedents),
                                                print_tags(consequents),
                                                print_tags(hidden))

for idx, (state, input) in enumerate(sa_pairs):
    print("- - - - - - - - -\nStep: {}".format(idx))
    print(print_state(state))
    print(input[1])
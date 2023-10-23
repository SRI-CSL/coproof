"""
Run through an entire proofdump and process it.

This is an extremely hacky attempt, and should be corrected to be a bit
more correct.
"""
from pathlib import Path
from pprint import pprint

from coprover import PROJ_ROOT
from coprover.feats.parsing.prfdmp import process
DATA_ROOT = Path(PROJ_ROOT, "data", "pvs", "konig")

accum = {}
for prf_file in DATA_ROOT.glob("*.prfdmp"):
    print(prf_file)
    sa_pairs = process(prf_file)
    accum[prf_file.name] = sa_pairs

for name, sa_pairs in accum.items():
    print("{}\t{}".format(name, len(sa_pairs)))
    for i, sa_pair in enumerate(sa_pairs):
        print("\t{}: {}".format(i, sa_pair.command()))

print(sa_pair.state['proofstate']['current-goal']['sequent']['consequents'][1])

sa_pairs = accum['konig-inf_path_build_seq.prfdmp']
cg_conseq = sa_pairs[0].state['proofstate']['current-goal']['sequent']['consequents']

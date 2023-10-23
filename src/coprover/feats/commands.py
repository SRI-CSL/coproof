"""
Routines for working with and predicting command predicates.

Because some vocab models may be subword based, not all returns
from a Transformers model may match exactly with a command label.

For example, it's entirely possible a hypothesized command to be
'exists' even though it was not in the original vocabulary.

This is used to go through and perform a prefix based match against
the known commands, by frequency of occurrence.  If no match is found,
the last character is removed and analysis continues until just the
baseline max-class guess is given.

Following the 'exists' example, this will match against 'existence-tcc'

The main entry point for this module is normalize().  Given a candidate
command, this will return back the best matching command from the available
set of commands.  This will also remove duplicates, returning the highest
canonical command to use.
"""


def normalize(prefix):
    """
    Given the command, attempts to match it as a prefix against the frequency ordered
    listing of commands.
    :param prefix:
    :return:
    """
    for cmd, _ in CMD_PRED_FREQS:
        if cmd.startswith(prefix):
            return _normalize_cmd(cmd)
    return normalize(prefix[0:-1])
    # Retain fallback to raise an exception as a sanity check
    raise Exception("Unknown command prefix={}".format(prefix))


def _normalize_cmd(cmd):
    """
    Normalizes duplicate commands to canonical form.
    :param cmd:
    :return:
    """
    if cmd == "instantiate":
        return "inst"
    return cmd


# Command predicates (based off of training from 220325_t5, derived from CVSLib)
CMD_PRED_FREQS = [('simplify', 1436),
                  ('assert', 1436),
                  ('expand', 1242),
                  ('inst', 1114),
                  ('lemma', 1102),
                  ('skolem', 871),
                  ('grind', 823),
                  ('skolem!', 610),
                  ('skosimp', 448),
                  ('skosimp*', 406),
                  ('split', 386),
                  ('instantiate', 357),
                  ('inst?', 352),
                  ('subtype-tcc', 283),
                  ('hide', 269),
                  ('flatten-disjunct', 251),
                  ('flatten', 251),
                  ('rewrite', 251),
                  ('typepred', 227),
                  ('prop', 205),
                  ('case', 195),
                  ('replace', 194),
                  ('ground', 180),
                  ('use', 161),
                  ('apply-extensionality', 156),
                  ('skolem-typepred', 125),
                  ('lift-if', 118),
                  ('skeep', 114),
                  ('simple-induct', 80),
                  ('induct', 79),
                  ('inst-cp', 68),
                  ('tcc', 60),
                  ('expand1*', 58),
                  ('expand*', 56),
                  ('iff', 55),
                  ('auto-rewrite', 49),
                  ('name', 45),
                  ('name-replace', 37),
                  ('smash', 34),
                  ('auto-rewrite-theory', 34),
                  ('forward-chain', 33),
                  ('case-replace', 27),
                  ('induct-and-simplify', 24),
                  ('decompose-equality', 22),
                  ('hide-all-but', 21),
                  ('delete', 19),
                  ('termination-tcc', 11),
                  ('let-name-replace', 11),
                  ('grind-with-ext', 10),
                  ('reveal', 9),
                  ('beta', 8),
                  ('reduce', 6),
                  ('mult-by', 6),
                  ('both-sides', 6),
                  ('induct-and-rewrite', 5),
                  ('measure-induct+', 3),
                  ('replace*', 3),
                  ('tcc-bdd', 3),
                  ('existence-tcc', 3),
                  ('judgement-tcc', 2),
                  ('skeep*', 2),
                  ('bddsimp', 1),
                  ('stop-rewrite', 1),
                  ('copy', 1),
                  ('do-rewrite', 1),
                  ('apply', 1),
                  ('auto-rewrite-theories', 1)]

if __name__ == "__main__":
    print("skosimp* -> ", "skosimp*")
    print("instantiate -> ", normalize("instantiate"))
    print("exist -> ", normalize("exist"))
    print("exists -> ", normalize("exists"))
    print("expan -> ", normalize("expan"))

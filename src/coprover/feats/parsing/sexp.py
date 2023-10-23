"""
Implements enough of a s-expression process for reading in
sequents and other information for the project.

Parts of this were derived from https://rosettacode.org/wiki/S-expressions#Python
"""

import re
from collections import OrderedDict

def parse(str_sexp, **kwargs):
    """ Given the string form of the s-expression in question,
    processes it into the appropriate list/dict types."""
    return _process(_parse_sexp(str_sexp), **kwargs)

term_regex = r'''(?mx)
    \s*(?:
        (?P<brackl>\()|
        (?P<brackr>\))|
        (?P<num>\-?\d+\.\d+|\-?\d+)|
        (?P<sq>"[^"]*")|
        (?P<s>[^(^)\s]+)
       )'''

def _parse_sexp(sexp, debug=False):
    stack = []
    out = []
    if debug: print("%-6s %-14s %-44s %-s" % tuple("term value out stack".split()))
    for termtypes in re.finditer(term_regex, sexp):
        term, value = [(t, v) for t, v in termtypes.groupdict().items() if v][0]
        if debug: print("%-7s %-14s %-44r %-r" % (term, value, out, stack))
        if term == 'brackl':
            stack.append(out)
            out = []
        elif term == 'brackr':
            assert stack, "Trouble with nesting of brackets"
            tmpout, out = out, stack.pop(-1)
            out.append(tmpout)
        elif term == 'num':
            v = float(value)
            if v.is_integer(): v = int(v)
            out.append(v)
        elif term == 'sq':
            out.append(value[1:-1])
        elif term == 's':
            out.append(value)
        else:
            raise NotImplementedError("Error: %r" % (term, value))
    assert not stack, "Trouble with nesting of brackets"
    return out[0]


def is_kv_list(sexp):
    for tok in sexp:
        if isinstance(tok, str) and tok.startswith(":"):
            return True
    return False


def _process(sexp, is_cmd=False):
    """
    Process s-expressions list from parse(), converting this into Python elements
    best matching the
    :param sexp:
    :param is_cmd:
    :return:
    """
    if isinstance(sexp, str) or not(isinstance(sexp, list)):
        # Singleton
        return sexp
    if sexp[0] == 'list':
        # Elements begin after ':elements'
        idx = sexp.index(':elements')
        if sexp[(idx+1)] == "nil":
            return []
        list_elts = sexp[(idx+1):][0]
        return [_process(sub_sexp) for sub_sexp in list_elts]
    elif len(sexp) <= 2:
        # Is a pair
        return [_process(sub_sexp) for sub_sexp in sexp]  # No CAR CDR here
    #elif sexp[0] == 'tuple' or sexp[0] == 'skolem' or is_cmd:
    elif sexp[0] == 'tuple':
        # Treat as a straight list if this is a tuple, skolem declaration, or a command with arglist
        return [_process(sub_sexp) for sub_sexp in sexp[1:]]
    elif is_kv_list(sexp) and not(is_cmd):
        # Is a dict, follows key-value notation
        ret = OrderedDict()
        for idx in range(1, len(sexp), 2):
            key = sexp[idx][1:]
            value = _process(sexp[idx + 1])
            ret[key] = value
        return {sexp[0]: ret}
    else:
        # Default to list, processing each element directly
        return [_process(sub_sexp) for sub_sexp in sexp]


"""
Utilities for processing JSON based exports
"""

import json
from collections import OrderedDict

TAG = 'tag'
LABEL = 'label'
DEPTH = "depth"

def _is_resv(key):
    return key in set([TAG, LABEL])

# Attempts at simplifying the JSON
class Tag:
    def __init__(self, tag, label=None, depth=None):
        self.tag = tag
        self.label = label
        self.elts = OrderedDict()
        self.depth = depth

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
        depth_str = " "
        if self.depth is not None:
            depth_str = " :depth {} ".format(self.depth)
        return "({} {}{}{})".format(self.tag, self.label,
                                    depth_str,
                                   elts_str).replace("\n", " ")

    def __str__(self):
        return self.pprint()


class Entries(Tag):
    def __init__(self):
        super(Entries, self).__init__(None, None)

    def pprint(self):
        elts_str = ""
        for k, v in self.elts.items():
            elts_str += " :{} {} ".format(k, str(v))
        return elts_str.replace("\n", " ")

    def __str__(self):
        return self.pprint()

def convert_dict2tags(exp):
    """
    Goes through the dict and converts them into Tags and Entries based based
    upon the dictionary fields.
    Args:
        exp: Python-native expression consisting of dicts and lists
    Returns:
        An Entries representation of the expression
    """
    if isinstance(exp, list):
        return [convert_dict2tags(x) for x in exp]
    elif isinstance(exp, str) or isinstance(exp, int) or exp is None:
        return exp
    assert isinstance(exp, dict)
    if 'tag' in exp:
        if DEPTH in exp:
            depth = exp[DEPTH]
        else:
            depth = None
        tag = Tag(exp[TAG], label=exp.get(LABEL, None), depth=depth)
        for k, v in exp.items():
            if not(_is_resv(k)):
                # Check to see if the value is an un-tagged type.  If so,
                tag.elts[k] = convert_dict2tags(v)
        return tag
    else:
        ret = Entries()
        for k, v in exp.items():
            ret.elts[k] = convert_dict2tags(v)
        return ret

"""
Experiments with parsing s-expressions sufficiently enough to featurize samples
"""


# Per https://rosettacode.org/wiki/S-expressions#Python
import re

dbg = False

term_regex = r'''(?mx)
    \s*(?:
        (?P<brackl>\()|
        (?P<brackr>\))|
        (?P<num>\-?\d+\.\d+|\-?\d+)|
        (?P<sq>"[^"]*")|
        (?P<s>[^(^)\s]+)
       )'''


def parse_sexp(sexp):
    stack = []
    out = []
    if dbg: print("%-6s %-14s %-44s %-s" % tuple("term value out stack".split()))
    for termtypes in re.finditer(term_regex, sexp):
        term, value = [(t, v) for t, v in termtypes.groupdict().items() if v][0]
        if dbg: print("%-7s %-14s %-44r %-r" % (term, value, out, stack))
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


def print_sexp(exp):
    out = ''
    if type(exp) == type([]):
        out += '(' + ' '.join(print_sexp(x) for x in exp) + ')'
    elif type(exp) == type('') and re.search(r'[\s()]', exp):
        out += '"%s"' % repr(exp)[1:-1].replace('"', '\"')
    else:
        out += '%s' % exp
    return out



sample = """
(proofstate :label "is_infinite_subtree_union" :current-goal
 (sequent :antecedents (list :elements nil) :consequents
  (list :elements
        ((s-formula :label nil :new? nil :formula
          (forall :bindings
           (list :elements
                 ((variable :type (type 801781668) :variable-name rr)
                  (variable :type (type 862779854) :variable-name tr)))
           :expression
           (apply :operator
                  (constant :constant-name booleans__IMPLIES :actuals
                   (list :elements nil) :type (type 1120869698))
                  :argument
                  (tuple
                   (apply :operator
                          (constant :constant-name booleans__NOT
                           :actuals (list :elements nil) :type
                           (type 3548206366))
                          :argument
                          (apply :operator
                                 (constant
                                  :constant-name
                                  finite_sets__is_finite
                                  :actuals
                                  (list
                                   :elements
                                   ((type-actual
                                     :type
                                     (type 801781668))))
                                  :type
                                  (type 2460366760))
                                 :argument
                                 (apply
                                  :operator
                                  (apply
                                   :operator
                                   (constant
                                    :constant-name
                                    konig__subtree
                                    :actuals
                                    (list :elements nil)
                                    :type
                                    (type 4191903916))
                                   :argument
                                   (variable
                                    :type
                                    (type 862779854)
                                    :variable-name
                                    tr))
                                  :argument
                                  (variable
                                   :type
                                   (type 801781668)
                                   :variable-name
                                   rr))))
                   (apply :operator
                          (constant :constant-name booleans__OR
                           :actuals (list :elements nil) :type
                           (type 1120869698))
                          :argument
                          (tuple
                           (apply :operator
                                  (constant
                                   :constant-name
                                   booleans__NOT
                                   :actuals
                                   (list :elements nil)
                                   :type
                                   (type 3548206366))
                                  :argument
                                  (apply
                                   :operator
                                   (constant
                                    :constant-name
                                    finite_sets__is_finite
                                    :actuals
                                    (list
                                     :elements
                                     ((type-actual
                                       :type
                                       (type 801781668))))
                                    :type
                                    (type 2460366760))
                                   :argument
                                   (apply
                                    :operator
                                    (apply
                                     :operator
                                     (constant
                                      :constant-name
                                      konig__subtree
                                      :actuals
                                      (list :elements nil)
                                      :type
                                      (type 4191903916))
                                     :argument
                                     (variable
                                      :type
                                      (type 862779854)
                                      :variable-name
                                      tr))
                                    :argument
                                    (apply
                                     :operator
                                     (constant
                                      :constant-name
                                      more_finseq__add
                                      :actuals
                                      (list
                                       :elements
                                       ((type-actual
                                         :type
                                         (type 1573404743))))
                                      :type
                                      (type 3660484718))
                                     :argument
                                     (tuple
                                      (integer :integer-value 0)
                                      (variable
                                       :type
                                       (type 801781668)
                                       :variable-name
                                       rr))))))
                           (apply :operator
                                  (constant
                                   :constant-name
                                   booleans__NOT
                                   :actuals
                                   (list :elements nil)
                                   :type
                                   (type 3548206366))
                                  :argument
                                  (apply
                                   :operator
                                   (constant
                                    :constant-name
                                    finite_sets__is_finite
                                    :actuals
                                    (list
                                     :elements
                                     ((type-actual
                                       :type
                                       (type 801781668))))
                                    :type
                                    (type 2460366760))
                                   :argument
                                   (apply
                                    :operator
                                    (apply
                                     :operator
                                     (constant
                                      :constant-name
                                      konig__subtree
                                      :actuals
                                      (list :elements nil)
                                      :type
                                      (type 4191903916))
                                     :argument
                                     (variable
                                      :type
                                      (type 862779854)
                                      :variable-name
                                      tr))
                                    :argument
                                    (apply
                                     :operator
                                     (constant
                                      :constant-name
                                      more_finseq__add
                                      :actuals
                                      (list
                                       :elements
                                       ((type-actual
                                         :type
                                         (type 1573404743))))
                                      :type
                                      (type 3660484718))
                                     :argument
                                     (tuple
                                      (integer :integer-value 1)
                                      (variable
                                       :type
                                       (type 801781668)
                                       :variable-name
                                       rr)))))))))))
          :asserted? nil)))
  :hidden (list :elements nil))
 :current-rule
 (list :elements
       (skolem 1 (list :elements ("rr" "tr")) (list :elements nil)))
 :current-input (list :elements (skeep))) ::: (skolem 1 ("rr" "tr") nil)"""

parsed = parse_sexp(sample)

from collections import OrderedDict

def process(sexp):
    if not(isinstance(sexp, list)):
        # Singleton
        return sexp
    if sexp[0] == 'list':
        # Elements begin after ':elements'
        idx = sexp.index(':elements')
        return [process(sub_sexp) for sub_sexp in sexp[(idx + 1):]]
    elif len(sexp) <= 2:
        # Is a pair
        return [process(sub_sexp) for sub_sexp in sexp]  # No CAR CDR here
    elif sexp[0] == 'tuple' or sexp[0] == 'skolem':
        return [process(sub_sexp) for sub_sexp in sexp[1:]]
    else:
        # Is a dict, follows
        ret = OrderedDict()
        for idx in range(1, len(sexp), 2):
            key = sexp[idx][1:]
            value = process(sexp[idx + 1])
            ret[key] = value
        return {sexp[0]: ret}

prf = process(parsed)
print(prf)

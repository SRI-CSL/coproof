import json
import pdb

from coprover.feats.featurize_cmdpred import ANT_TOK, CONS_TOK, HIDDEN_TOK, NOOP_CMD

PROOF_SESSION = "proof-session"
PROOF = "proof"
TYPE_HASH = "type-hash"

MODULE_WITH_HASH = "module-with-hash"
MODULE_WITH_HASH2 = "moduleWithHash"
MODULE = "module"
THEORY = "theory"
CONST_DECL = "const-decl"
CONST_DEF = "const-def"
DECLARATIONS = "declarations"
DEFINITION = "definition"
FORMULA_DECL = "formula-decl"
FORMULA_DECL2 = "formulaDecl"

ID = "id"
LAMBDA = "lambda"
LEMMA = "LEMMA"
NAME = "name"

TYPE_HASH = "type-hash"
TYPE_HASH2 = "typeHash"

ENTRIES = "entries"
TYPENAME = "typename"
TUPLETYPE = "tupletype"
FUNCTIONTYPE = "functiontype"
SUBTYPE = "subtype"

TAG = "tag"
TUPLE = "tuple"
LABEL = "label"
SEQUENT = "sequent"
CURRENT_GOAL = "current-goal"
ANTECEDENTS = ANT_TOK
CONSEQUENTS = CONS_TOK

#ANTECEDENTS = "antecedents"
#CONSEQUENTS = "consequents"

PROOFSTATE = "proofstate"

ACTUALS = "actuals"
ARGUMENT = "argument"
ARGUMENTS = "arguments"
APPLY = "apply"
ASSERTED = "asserted?"
CONSTANT = "constant"
CONSTANT_NAME = "constant-name"
EXPRESSION = "expression"
FIELD = "field"
FORALL = "forall"
FOREACH = "foreach"
GETFIELD = "getfield"
INTEGER = "integer"
INTEGER_VALUE = "integer-value"
VARIABLE_NAME = "variable-name"

FORMULA = "formula"
SFORMULA = "s-formula"

OPERATOR = "operator"
TYPE = "type"

TRUE = "True"
FALSE = "False"
IS_ASSERTED = "isasserted"
NOT_ASSERTED = "notasserted"

INTEGER_PLACEHOLDER = "INT"

CURRENT_RULE = "current-rule"
CURRENT_INPUT = "current-input"

RULE = "rule"

from collections import Counter
# Try again, this time using a strict traversal collecting tags


def bfs_flatten(iterable):
    """ Performs a breadth-first-search flattening of a tree into a list """
    ret = []
    for x in iterable:
        if isinstance(x, list):
            ret.extend(bfs_flatten(x))
        elif x is None:
            continue
        else:
            ret.append(x)
    return ret


TABU_TAGS = set([TAG, ACTUALS, CONSTANT_NAME, LABEL,
                 TYPE, VARIABLE_NAME, CURRENT_RULE, CURRENT_INPUT])


def safe_get(elt, *keys):
    """ Does an element access with the key in regular and camel form """
    all_keys_to_consider = []
    for key in keys:
        all_keys_to_consider.append(key)
        # Add camel case form of the key
        toks = key.split("-")
        camel_form = "".join( toks[0:1] + [x.capitalize() for x in toks[1:]])
        all_keys_to_consider.append(camel_form)
    for key in all_keys_to_consider:
        if key in elt:
            return elt[key]
    raise Exception("Unable to find key={}".format(key))

    
def process(elt, parent_key=None, type_lookup={}):
    """
    Collect all tags using a strict traversal.  If this is a constant, check
    the context and if is an operator, collect the constant name
    
    TODOS: Implement multiple policies for featurizing variables
    TODOS: Use type hash to convert into their appropriate types
    """
    accum = []
    if isinstance(elt, list):
        # Excise (tag, tuple) which appears to be boilerplate
        if len(elt) == 2 and elt[0] == 'tag' and elt[1] == 'tuple':
            return [None]
        return [process(x, type_lookup=type_lookup) for x in elt]
    elif isinstance(elt, str):
        return [elt]
    elif isinstance(elt, int):
        return [elt]
    elif isinstance(elt, dict):
        if TAG in elt:
            elt_tag = elt[TAG]
            if isinstance(elt_tag, list):
                # accum.extend(elt_tag)
                elt_tag = elt_tag[0] # TODO: account for multi-item element tag
            #accum.append(elt_tag)  # Removed default addition of tags, to reduce verbosity
            if elt_tag == CONSTANT and parent_key == OPERATOR:
                accum.extend(process(safe_get(elt, CONSTANT_NAME, ID), type_lookup=type_lookup))
            for k, v in elt.items():
                if k not in TABU_TAGS:
                    accum.append(k)
                    accum.extend(process(v, parent_key=k, type_lookup=type_lookup))
            if VARIABLE_NAME in elt and TYPE in elt:
                # Grab and replace the type from lookup
                elt_type = str(safe_get(elt, TYPE))
                if elt_type in type_lookup:
                    accum.append(type_lookup[elt_type])
                else:
                    accum.append("UnkType") # Placeholder for now
    elif elt is None:
        return [ "null" ]
                
    return bfs_flatten(accum)



def read_typehash(root_obj):
    assert TYPE_HASH in root_obj or TYPE_HASH2 in root_obj
    if TYPE_HASH in root_obj:
        typehash_root = root_obj[TYPE_HASH]
    else:
        typehash_root = root_obj[TYPE_HASH2]
    if isinstance(typehash_root, dict):
        elt = typehash_root[TAG]
        if isinstance(elt, str):
            assert elt == "typelist"
        else:
            assert elt[0] == "typelist"
    elif isinstance(typehash_root, list):
        return {} # Empty type listing 
    type_lookup = {}
    # Do a first pass to collect basic types, then do a second
    # pass to collect those operating off those types.
    # Only do two passes for now, and worry about more complicated
    # typing structures later.
    for type_ptr, type_entry in typehash_root[ENTRIES].items():
        if type_ptr == FUNCTIONTYPE:
            return FUNCTIONTYPE  # Unusual, is Lisp code
        if isinstance(type_entry, str):
            return FUNCTIONTYPE  # Unusual, is Lisp code
        type_tag = type_entry[TAG]
        if type_tag == TYPENAME:
            type_lookup[type_ptr] = type_entry[ID]
        else:
            type_lookup[type_ptr] = type_tag
    return type_lookup


def read_theory(json_root):
    if not(isinstance(json_root, dict)):
        f = open(json_root, 'r')
        json_root = json.load(f)
        f.close()
    assert json_root[TAG] == MODULE_WITH_HASH or json_root[TAG] == MODULE_WITH_HASH2 
    type_lookup = read_typehash(json_root)
    theory_lookup = {}
    module_list = json_root[MODULE]
    # Account for tendency to replace singleton lists with just the item in JSON
    if isinstance(module_list, dict):
        module_list = [module_list]
    for module in module_list:
        if module[TAG] == THEORY:
            theory_id = module[ID]
            for decl in module[DECLARATIONS]:
                # lamdas
                if decl[TAG] == CONST_DECL and \
                  CONST_DEF in decl and \
                  decl[CONST_DEF] is not None and TAG in decl[CONST_DEF] and \
                  decl[CONST_DEF][TAG] == LAMBDA:
                    name = safe_get(decl, NAME, ID)
                    expression = process(decl[CONST_DEF][EXPRESSION], type_lookup=type_lookup)
                    theory_lookup[name] = bfs_flatten(expression)
                elif decl[TAG] == FORMULA_DECL or decl[TAG] == FORMULA_DECL2:
                    # Encompasses obligations and lemmas
                    name = decl[ID]
                    expression = process(decl[DEFINITION])
                    theory_lookup[name] = bfs_flatten(expression)
    return theory_lookup


def read_proof_session(json_root):
    if not(isinstance(json_root, dict)):
        f = open(json_root, 'r')
        json_root = json.load(f)
        f.close()
    assert PROOF in json_root and json_root[TAG] == PROOF_SESSION
    proof_seq = json_root[PROOF]
    sa_pairs = []
    if proof_seq is not None:
        type_lookup = read_typehash(json_root)
        for idx in range(0, len(proof_seq), 2):
            proof_step = proof_seq[idx]
            input_step = proof_seq[idx + 1]
            state_toks = bfs_flatten(process(proof_step, type_lookup=type_lookup))
            if isinstance(input_step, list):
                input_cmd = input_step[1]
            elif isinstance(input_step, dict):
                input_cmd = input_step[RULE]
            sa_pairs.append((state_toks, input_cmd))
    return sa_pairs


def is_proof_json(json_root):
    return PROOF in json_root and json_root[TAG] == PROOF_SESSION


def read_proof_session_lemmas(json_root):
    """
    Given the JSON root object of a PVS proof session, featurizes the state and returns
    the state tokens, input command, and the command arguments.
    
    NOTE: This currently does a tree based flattening, and does not distinguish the
    antecedent, consequent, and hidden states (treating them directly as tags)
    """
    if not(isinstance(json_root, dict)):
        f = open(json_root, 'r')
        json_root = json.load(f)
        f.close()
    if not(is_proof_json(json_root)):
        return None
    proof_seq = json_root[PROOF]
    sa_tuples = []
    if proof_seq is not None:
        type_lookup = read_typehash(json_root)
        for idx in range(0, len(proof_seq), 2):
            proof_step = proof_seq[idx]
            input_step = proof_seq[idx + 1]
            state_toks = bfs_flatten(process(proof_step, type_lookup=type_lookup))
            if isinstance(input_step, list):
                input_cmd = input_step[1]
                input_arg = input_step[2]
            elif isinstance(input_step, dict):
                input_cmd = input_step[RULE]
                input_arg = input_step[ARGUMENTS]
            # Get the first arg
            if input_arg is not None:
                if isinstance(input_arg, list):
                    input_arg = input_arg[0]
            sa_tuples.append((state_toks, input_cmd, input_arg))
    return sa_tuples

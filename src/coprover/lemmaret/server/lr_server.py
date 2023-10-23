from flask import Flask
from flask import request, jsonify
import numpy as np
import pdb

from coprover.lemmaret.featurizer import process_formula_json, bfs_flatten, TAG, CONST_DECL, CONST_DEF, EXPRESSION, LAMBDA, FORMULA_DECL, FORMULA_DECL2, ID, DEFINITION
from coprover.lemmaret.theorybank import gen_sbert_theorybank
from coprover.lemmaret.featurizer import safe_get

POST = "POST"
GET = "GET"
QUERY = "query"
STATE = "state"
NAME = "name"

FORMULA1 = "formula1"
FORMULA2 = "formula2"


def extract_formula(decl):
    """Given a JSON expression, either a formula declaration or a sequent
    state, returns the JSONified form suitable as input for the system."""
    if TAG in decl:
        # 'tag' is a key, so is from theory
        if decl[TAG] == CONST_DECL and \
            CONST_DEF in decl and \
                decl[CONST_DEF] is not None and TAG in decl[CONST_DEF] and \
                decl[CONST_DEF][TAG] == LAMBDA:
            name = safe_get(decl, NAME, ID)
            expression = process_formula_json(decl[CONST_DEF][EXPRESSION], type_lookup=type_lookup, ret_str=False)
        elif decl[TAG] == FORMULA_DECL or decl[TAG] == FORMULA_DECL2:
            # Encompasses obligations and lemmas
            name = decl[ID]
            expression = process_formula_json(decl[DEFINITION])
    else:
        # Otherwise is a state expression
        expression = process_formula_json(decl[STATE])
    expression = bfs_flatten(expression)        
    return " ".join([str(x) for x in expression])


class LemmaRetrievalServer:
    def __init__(self, top_N=5):
        self.theorybank = gen_sbert_theorybank()
        self.top_N=top_N

    def hello_world(self):
        return "Hello"

    def query(self):
        """ Given a sequent state from the proofstate trace, returns the top 5 relevant results
        in order.
        """
        if request.method == POST:
            json_ = request.get_json(force=True)
            query = json_[STATE]
        elif request.method == GET:
            query = request.args.get(QUERY)[STATE]
        else:
            raise Exception("Unsupported request method={}!".format(request.method))
        tok_query = process_formula_json(query, ret_str=True)
        results = self.theorybank.query(tok_query, top_N=self.top_N, return_scores=True)
        return jsonify(results)

    def compare(self):
        """ Given two formulas, either a lemma or a sequent state, and returns their relevance to each other.
        """
        if request.method == POST:
            json_ = request.get_json(force=True)
            formula1 = json_[FORMULA1]
            formula2 = json_[FORMULA2]
            tok_form1 = extract_formula(formula1)
            tok_form2 = extract_formula(formula2)
            q1 = self.theorybank.vectorize(tok_form1)
            q2 = self.theorybank.vectorize(tok_form2)
            sim = np.dot(q1, q2.transpose())
            results = {
                "score": float(sim[0, 0])
                }
            return jsonify(results)
        else:
            raise Exception("Unsupported request method={}!".format(request.method))
        

    def add_routes(self, app):
        app.add_url_rule("/", "hello_world", self.hello_world, methods=["GET", "POST"])
        app.add_url_rule("/query", "query", self.query, methods=["GET", "POST"])
        app.add_url_rule("/compare", "compare", self.compare, methods=["POST"])        
        return app


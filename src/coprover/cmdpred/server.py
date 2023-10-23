"""
Flask based frontend for Command Prediction
"""

from flask import Flask
from flask import request, jsonify

from coprover.cmdpred import CmdPredictor
from coprover.feats.featurize_cmdpred import format_state

POST = "POST"
GET = "GET"

STATE = "state"
CMD_HISTORY = "cmd_history"

class CmdPredServer:
    def __init__(self, use_gpu=True, use_device="cuda"):
        self.cmdpred = CmdPredictor(use_gpu=use_gpu, use_device=use_device)
        print(self.cmdpred)

    def hello_world(self):
        return "<p>Yoooooo</p>"

    def query(self):
        if request.method == POST:
            json_ = request.get_json(force=True)
            state = json_[STATE]
            cmd_history = json_[CMD_HISTORY]
        elif request.method == GET:
            state = request.args.get(STATE)
            cmd_history = request.args.get(CMD_HISTORY)
        else:
            raise Exception("Unsupported request method={}!".format(request.method))
        results = self.cmdpred.predict(state, cmd_history)
        return jsonify(results)

    def add_routes(self, app):
        app.add_url_rule('/', 'hello_world', self.hello_world, methods=["GET", "POST"])
        app.add_url_rule('/query', 'query', self.query, methods=["GET", "POST"])
        return app
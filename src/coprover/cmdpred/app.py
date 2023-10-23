from flask import Flask
from coprover.cmdpred.server import CmdPredServer

USE_GPU=True
USE_DEVICE="cuda"

cmdpred_server = CmdPredServer(use_gpu=USE_GPU, use_device=USE_DEVICE)
app = cmdpred_server.add_routes(Flask(__name__))

def start_service():
    app.run(host='0.0.0.0', debug=False, port=7001)

if __name__ == "__main__":
    start_service()
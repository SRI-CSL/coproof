from flask import Flask
from coprover.lemmaret.server.lr_server import LemmaRetrievalServer

TOP_N_RESULTS = 100
print("Starting Lemma Retrieval service, top_N={}".format(TOP_N_RESULTS))

lemma_retrieval_server = LemmaRetrievalServer(top_N=TOP_N_RESULTS)
app = lemma_retrieval_server.add_routes(Flask(__name__))

def start_service():
    app.run(host='0.0.0.0', debug=False, port=7111)

if __name__ == "__main__":
    start_service()

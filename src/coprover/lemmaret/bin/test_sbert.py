from pathlib import Path
import json

from coprover.lemmaret.theorybank import gen_sbert_theorybank
from coprover import RSC_ROOT

with open(Path(RSC_ROOT, "lemma_retrieval", "lemma_requests.json"), "r") as f:
    lemma_requests = json.load(f)


theorybank = gen_sbert_theorybank()

lemma_request = lemma_requests[0]
sequent = " ".join(lemma_request['state'])
gold = lemma_request['name']
titles = theorybank.query([sequent], top_N=5)
print(gold in titles)
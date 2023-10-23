# CoProver
CoProver is a proof recommendation system for proof assistants trained on PVS and Imandra

## Installation

CoProver is intended to run on Python 3.9+.  Once an environment has been selected, install the dependencies via `setup.py`

```
cd coproof
pip install -e .
```

## Tasks

### Training and running the PVS Command Predictor experiments

To train and evaluate the PVS command prediction, execute this module,

```
python -m coprover.cmdpred.train_t5
```

Note this uses PyTorch Lightning to do DDP training across several GPUs.  You may need to adjust the `num_gpus` argument to match the number of GPUs you have on your system, or use 0 for software only.

The baselines for comparison can be executed via,

```
python -m coprover.cmdpred.run_baselines
```

### Training and running the PVS Lemma Retrieval experiments

To first train the underlying Sentence BERT model, 

```
python -m coprover.lemmaret.bin.train_sbert
```

### Generating JSON formatted data from PVS JSON proof traces
python -m coprover.feats.featurize_cmdpred
Experiments using T5 to perform pre-training tasks 
and featurize for the Co-Prover task.

Most of these experiments are conducted using the SimpleT5 library,

  https://github.com/Shivanandroy/simpleT5

Notes related to these sets of experiments will be listed
here, organized by date.

Most of the experiments here rely on the JSON tag based featurization
of the proofs under data/pvs/prelude.  This is done by executing,

   python 220325_data_setup.py

## Notes

### 220331

MLM using single masked token evaluation trained on Prelude.

Trained for 9 epochs, with corrupting N=1 tokens per sentence

Test acc at N=1,

	 mu/std=0.851/0.230, min/max=0.000/0.991

Test acc at N=2




### 220325

SimpleT5 works, but need to have > 1 tokens predicted for target (just
1 leaves blank guesses) Need to have a command prefix with a colon,
otherwise training will not work

Using the tag labels as features and targeting the command predicate,
we get the following accuracies, on the prelude:

Train acc=0.603
Test acc=0.325

This is against a baseline maxclass accuracy guess of 0.16 (train),
0.17 (test).

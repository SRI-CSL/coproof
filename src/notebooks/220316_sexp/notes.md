# 220322

## Same state, different commands
Tag based notation, Markov violated between step 0 and step 1, and then steps 2 and 3.  Same state, different commands.  


FILE=/Users/yeh/proj/CoProver/src/coprover/../../data/pvs/prelude/integers-proofs/minus_int_is_int.json
Step: 0

|----
  s-formula forall  ['variable'] apply constant apply constant ['variable']
Hidden:
CMD: skolem

Step: 1

|----
  s-formula forall  ['variable'] apply constant apply constant ['variable']
Hidden:
CMD: skeep

Step: 2
  s-formula apply constant apply constant constant
|----
  s-formula apply constant apply constant constant
Hidden:
CMD: lemma

Step: 3
  s-formula apply constant apply constant constant
|----
  s-formula apply constant apply constant constant
Hidden:
CMD: rewrite
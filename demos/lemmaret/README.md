# Lemma Retrieval Server

## Installation

For these instructions, paths are relative to the directory CoProver
is checked out in.

This requires Python 3.6+ to work.

Note, you may wish to do the installation in a custom pyenv or conda
environment.

1. Download and install the latest version of CoProver:

   git clone https://github.com/SRI-CSL/CoProver.git
   cd CoProver/src
   pip install -e .
  
2. Download the minimal lemma retrieval models and resources.

   Download the file lemma_retrieval_minimal.tgz archive from SRI's sharepoint:

      https://sriintl.sharepoint.us/:u:/r/sites/ics_darpa_pearls/Shared%20Documents/Development/ML/models/lemma_retrieval_minimal.tgz?csf=1&web=1&e=gkVDfx

   Move the file to the resources directory and unarchive it:

      mv lemma_retrieval_minimal.tgz CoProver/resources
      cd CoProver/resources
      tar xvzf lemma_retrieval_minimal.tgz

## Demo Usage

1. Start the lemma retrieval service by entering the following on the
   CLI:

       lemmaret_server

   This will start the lemma retrieval Flask service on port 7111 of
   localhost.

   To change the number of top lemmas retrieved, change the following line in src/coprover/lemmaret/server/app.py, line 4:

      TOP_N_RESULTS = 10

2. Execute the demo query:

       cd CoProver/demos/lemmaret
       ./demo_query.sh

   This will use Curl to push the JSON query in
   CoProver/demos/lemmaret/demo_state.json to the service.  Note that
   this is the same demo_state.json as used for the command predicate
   prediction task.

   The JSON payload is exactly the same, except this calls port 7111
   with the query path, e.g.,

        https://localhost:7111/query

3. Execute the state and lemma relevance scorer:

       cd CoProver/demos/lemmaret
       ./demo_compare.sh

   This references the file demo_compare.json, which describes how the
   query payload is to be formatted.  The query is a dictionary with
   two keys, "formula1" and "formula2".  One references the state
   representation, the other the lemma declaration.  Note that the
   ordering does not matter.

   This calls port 711 with the compare path:

        https://localhost:7111/compare

   Note that the state representation uses the same representation as
   the command prediction task's.  While it would be more efficient
   just to use the state definition, we currently use the same format
   to ensure consistency, but may change this if needed.

   The lemma formula is the portion from the theory where the tag
   indicates the lemma declaration, e.g., "formula-decl".  Note that
   currently this does not embed the typehash information, and we will
   include this in another version.

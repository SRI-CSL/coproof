# Command Prediction service

This illustrates how to set up and query the CoProver PVS Command
Prediction service.

For these instructions, we presume $COPROVER_ROOT is the directory the
CoProver project was checked out in.

## Downloading the model

The trained model needs to be downloaded and installed.  First create the model directory:

    mkdir $COPROVER_ROOT/resources/pvs_cmd_pred/models

The model itself is can be downloaded from the DARPA PEARLS Sharepoint, at the path:

    Development > ML > models > models_220325_t5 > models

Please download the "cmd_pred1_hist3" directory, and then install it into

    $COPROVER_ROOT/resources/pvs_cmd_pred/models/.

Note, please be sure the curr_best softlink is active and pointing to
the directory simplet5-epoch-9-train-loss-0.2947-val-loss-0.3488,
e.g.,

    $COPROVER_ROOT/pvs_cmd_pred/models/cmd_pred1_hist3/
          curr_best -> simplet5-epoch-9-train-loss-0.2947-val-loss-0.3488

A convenience read-only share link is at,

    https://sriintl.sharepoint.us/:f:/s/ics_darpa_pearls/EkoKZR7NHcxPjTly0UVVSHkBuOt1yFZbuK3zJ7wpoeXzEg?e=eB4Cj6

## Python Setup and Environments
Ensure that Python 3.9+ is installed on your system.  We recommend
using the Miniconda distribution, and we present instructions for
downloading and setting up the environment here.

Please download and run the appropriate installer from here,

    https://docs.conda.io/en/latest/miniconda.html

Once installed, create a Python 3.9 environment.  Environments are
full Python installations with necessary libraries, and the conda tool
is used to activate or deactivate environments.  When an environment
is activated, that version of Python and installed libraries are used.
Note while multiple environments can be active at a time, we recommend
only one be active.

To create the environment, please enter the following on the CLI,

    conda create --name coprover python=3.9

Conda adjusts the shell so the current active Python environment name is displayed.  For example, on my laptop when the coprover environment is active, the environment name prefixes the rest of the shell,

    (coprover) yeh@Kirk ~ %

To activate the coprover environment, enter,

    conda activate coprover

To deactivate the current environment,

    conda deactivate

For CoProver, we recommend only the coprover environment be active.

## Command Prediction Service

Please make sure have the coprover Python environment installed and
activated prior to performing these steps.

Please execute the following to install and start the service,

    cd $COPROVER_ROOT/src
    pip install -e .
    cmdpred_server

This will start up the server on the localhost at port 7001.

## Demonstration Script

The script demo_query.sh demonstrates how to query the service.  The
script uses curl to post the contents of the file demo_state.json to
the service at http://localhost:7001/query.

Command prediction queries should be POSTed to
http://localhost:7001/query.  The POST body content is a JSON dictionary formatted as follows,

The "state" key points to a proof-state step in JSON form.  Note that
"current_input" is not required, and was used in data development for
generating the ground truth actions for the training data.

The "cmd_history" key points to a list showing the last three commands
issued by the user, with the most recent command being on the right.
For example, if the user started with "grind," then "simplify," and
finally "skolem," the cmd_history value would be

    "cmd_history": ["grind", "simplify", "skolem"]

Note that initially the the history consists of "NOOP" special
commands, indicating the lack of a command for that portion of the
history.

The server returns the top-5 results as a list, with the higher ranked
predictions on the left.  For the above example, we'd get,

    ["expand","lemma","typepred","grind","simplify"]

With "expand" being the top-most prediction.
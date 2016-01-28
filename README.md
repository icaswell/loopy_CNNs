Overview of the folders:
========================


architectures
-------------
A folder defining different architectures used by our models.  Contains:
 + architecture_config_asserter.py: a script to may sure no silly mistakes have been made in specifying the architecture
 + architecture_config_readme.py: a file explaining how to make architecture files and assert they are feasible.


code
------
 + *extraneous_scripts*: a folder of whatever random scripts we may create for, say, plotting graphs, cleaning data, etc.
 + *util.py*:  Contains some super clutch functions to do things like print in color and make really descriptive filenames. Highly recommended you run python util.py


data
-----
Self explanatory


saved_models
------------
This contains various checkpoints and saved models from during the training process.  They contain saved parameters and information about the run (how long it took to train, when it was trained, etc.), and a link to the architecture config file


writeup
---------
our report and everyhing related to it.   Though actually, maybe it's better just ot use overleaf.

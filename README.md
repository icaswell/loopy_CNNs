Overview of the folders:
========================


architectures
-------------
A folder defining different architectures used by our models.  Contains:
 + architecture_config_asserter.py: a script to may sure no silly mistakes have been made in specifying the architecture
 + architecture_config_readme.py: a file explaining how to make architecture files and assert they are feasible.

code
------
Contains the central scripts and models to our project.  All scripts have a demo that can be run by running them as __main__.  These include:
 + **loopy_cnn.py**: defines the class LoopyCNN(), that will be the main character of this project. Note that this is a general model, so if e.g. we decide at the last minute to do ResNet, it'll take nothing but a new architecture config file!
 + **util.py**:  Contains some super clutch functions to do things like print in color and make really descriptive filenames. Highly recommended you run python util.py
 + **extraneous_scripts**: a folder of whatever random scripts we may create for, say, plotting graphs, cleaning data, etc.

data
-----
Self explanatory

results
----
A series of dated text files concisely summarizing results from different runs

saved_models
------------
This contains various checkpoints and saved models from during the training process.  They contain saved parameters and information about the run (how long it took to train, when it was trained, etc.), and a link to the architecture config file

writeup
---------
our report and everyhing related to it.   Though actually, maybe it's better just ot use overleaf.

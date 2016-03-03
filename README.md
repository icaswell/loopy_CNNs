Authors:
--------
Isaac Caswell, Chuanqi Shen, Lisa Wang

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
 + **abstract_loopy_cnn.py**: defines the class AbstractLoopyCNN().  This reads in a config file (like those in the architectures directory) and constructs a network.  Note that this is an abstract class and must be inherited by a framework-specific class, such as loopy_cnn_lasagne.py
 + **loopy_cnn_lasagne.py**: defines the class LoopyCNN() as implemented in lasagne, that will be the main character of this project. Note that this is a general model, so if e.g. we decide at the last minute to do ResNet, it'll take nothing but a new architecture config file! 
 + **util.py**:  Contains some super clutch functions to do things like print in color and make really descriptive filenames. Highly recommended you run python util.py.
 + **batchnorm_layer.py**:  Defines a spatial batchnorm layer.   Taken largely unmodified from Jan Schlüter's implementation
 + **guided_backprop.py**:  exports a function to take a trained LoopyCnn model and an image, perform guided backprop with respect to a user-defined set of convolutional filters. The result is saved to a timestamped filesystem with one folder per layer, and within each of those one folder per filter, each of which contains one .png file for each input  image.  The original input images are saved in a subdirectory called "orig".
 + **guided_backprop_util.py**:  util functions for guided_backprop.py.  Modified from saliency map scripts from Jan Schlüter.
 + **image_utils.py**: utility functions for plotting images.
 + **extraneous_scripts**: a folder of whatever random scripts we may create for, say, plotting graphs, cleaning data, etc.

data
-----
Self explanatory

pictures
-----
results of runs, say, of guided backprop.

results
----
A series of dated text files concisely summarizing results from different runs, as well as a folder containing plots of loss over time for different runs

saved_models
------------
This contains various checkpoints and saved models from during the training process.  They contain saved parameters and information about the run (how long it took to train, when it was trained, etc.), and a link to the architecture config file

shellscripts
------------
Contains a collection of shellscripts, e.g. to download data.

writeup
---------
our report and everyhing related to it.   Though actually, maybe it's better just ot use overleaf.

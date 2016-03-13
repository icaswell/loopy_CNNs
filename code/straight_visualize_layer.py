# File: guided_backprop.py
# @author: Isaac Caswell
# @created: 29 Feb 2016
#===============================================================================
# DESCRIPTION:
#
# exports a function to take a trained lasagne model and an image, perform guided
# backprop with respect to a user-defined set of convolutional filters.
# The result is saved to a timestamped filesystem with one folder per layer, and within each of 
# those one folder per filter, each of which contains one .png file for each input 
# image.  The original input images are saved in a subdirectory called "orig".
#
#===============================================================================
# TODOS:
# - go over all filters, not just first, per layer
#===============================================================================


import numpy as np
from collections import defaultdict
import sys, os
import time
from pprint import pprint
sys.path.append("../architectures")
import cPickle as pickle

import theano
import theano.tensor as T
import lasagne

from loopy_network_lasagne import LoopyNetwork
import util
from image_utils import visualize_image
from guided_backprop_util import *
from data_utils import *


#===============================================================================
# 


def save_attention_visualizations_to_folder(model, dirname, X,
											run_stem="attention_vis",
											dataset_name="cifar10"):
	"""
	uses the heuristic that composition nodes have the word "input" in them
	"""

	input_var = model.input_var
	names_to_layers = {layer.name:layer for layer in lasagne.layers.get_all_layers(model.network) if layer is not None and layer.name is not None}
	N = X.shape[0]
	# layer_id = 5
	
	run_folder_name = os.path.join(dirname, run_stem + "_" + util.time_string(precision="minute"))
	orig_folder_name = os.path.join(run_folder_name, "orig")	
	os.mkdir(run_folder_name) #make the folder in which to store images for this run
	os.mkdir(orig_folder_name) #make the folder for the original images
	for image_i, image in enumerate(X):
		saveto = os.path.join(orig_folder_name, "img_%s"%image_i)
		visualize_image(image, dataset=dataset_name, saveto=saveto)
	# layers_to_vis = [l.name for l in lasagne.layers.get_all_layers(model.network) if l.name and "input" in l.name]
	layers_visualized = 0
	for layer_name, internal_top_layer in names_to_layers.items():
		if "input" not in layer_name: continue
		layers_visualized += 1
		layer_folder_name = os.path.join(run_folder_name, "merge_%s"%layers_visualized)
		os.mkdir(layer_folder_name)
		print "\t visualizing layer %s in folder %s"%(layer_name, layer_folder_name)

		# output volume has shape (batch_size, channels, height, width)	
		output_volume = lasagne.layers.get_output(internal_top_layer, deterministic=True)

		attn_fn = theano.function([input_var], output_volume)
		attn_maps = attn_fn(X)
		print "output volume shape: ", output_volume.shape
		for image_i, image in enumerate(attn_maps):		
			saveto = os.path.join(layer_folder_name, str(image_i))
			visualize_image(image, dataset=dataset_name, saveto=saveto)

if __name__=="__main__":
	#===============================================================================
	# Sample script
	# X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

	# saved_model = "../saved_models/mnist_c3_c3_c1_fc+addition-loop_Feb-27-2016_epoch=19"
	# saved_model = "../saved_models/layers=5_loops=1_architecture-ID=10a222a5f3757ea7f2fa6cfafd3a514cdd22d8ca_Feb-20-2016_epoch=25"
	# arch_fpath = "../architectures/mnist_c3_c3_c1_fc+loop.py"
	X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()

	arch_fpath = "../architectures/cifar_isaac.py"
	saved_model = "../saved_models/deadweek_multiplication_loop_unrolls=3_Mar-13-12:05:13-2016_epoch=19"
	model = LoopyNetwork(architecture_fpath=arch_fpath, n_unrolls=5)
	model.load_model(saved_model)

	#===============================================================================
	# Now actually visualize
	save_attention_visualizations_to_folder(model, dirname="../pictures", 
												X=X_train[0:5],
												run_stem="test_attn_vis"
												)



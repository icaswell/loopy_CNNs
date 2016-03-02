# File: guided_backprop.py
# @author: Isaac Caswell
# @created: 29 Feb 2016
#===============================================================================
# DESCRIPTION:
#
# exports a function to take a trained lasagne model and an image, and then save 
# the result to a filesystem with one folder per filter.  
# Each file that is saved is numpy matrix, yet to be converted into an image by 
# someone else
#
#===============================================================================
# TODOS:
# -put model modification into the function
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
from guided_backprop_util import *
from data_utils import *


#===============================================================================
# 


def save_filter_visualizations_to_folder(model, dirname, X, layers_to_vis=None, 
														filters_to_vis={}, 
														run_stem="guided_backprop"):
	"""
	Uses guided backprop to visualize all filters in all relevant layers.  Makes a folder containing all these, and within 
	that folder makes a folder for each layer.

	:param list layers_to_vis: a list of layer names to visualize.  If it is None, all layers are visualized.
	:param list filters_to_vis: a dict mapping layers in layers_to_vis to indexes of filters to visualize.  
			If there is no entry for an entry in layers_to_vis, then all filters for that layer are visualized.
	:param string run_stem: the stem of the name of the folder for this run
	:param string dirname: the name of the directory in which to save this folder
	:param array-like X: images with respect to which to get the guided bp.  Shape N, C, H, W I think
	:param LoopyNetwork model: a trained lasagne model.
	"""
	input_var = model.input_var
	names_to_layers = {layer.name:layer for layer in lasagne.layers.get_all_layers(model.network) if layer is not None and layer.name is not None}
	N = X.shape[0]
	# layer_id = 5
	
	run_folder_name = os.path.join(dirname, run_stem + "_" + util.time_string(precision="second"))
	os.mkdir(run_folder_name) #make the folder in which to store images for this run

	if layers_to_vis is None:
		layers_to_vis = names_to_layers.keys()
	for layer_name, internal_top_layer in names_to_layers.iteritems():
		if layer_name not in layers_to_vis: continue
		layer_folder_name = os.path.join(run_folder_name, layer_name)
		os.mkdir(layer_folder_name)
		print "\t visualizing layer %s in folder %s"%(layer_name, layer_folder_name)

		# internal_top_layer = names_to_layers[layer_name]
		filters_to_vis_for_this_layer = filters_to_vis.get(layer_name, None)


		# output volume has shape (batch_size, channels, height, width)	
		output_volume = lasagne.layers.get_output(internal_top_layer, deterministic=True)

		if filters_to_vis_for_this_layer is None: # look at all filters in the layer if none is/are specified
			# filters_to_vis_for_this_layer = range(output_volume.shape[1])
			#TODO: how to access the number of filters per layer?  We want this to be the above line (with range())
			filters_to_vis_for_this_layer = [0]

		#===============================================================================
		# No go through and do guided backprop for each filter:
		for filter_i in filters_to_vis_for_this_layer:

			#===============================================================================
			# Get the slice of the output volume corresponding to this filter, then sum it.
			# Differentiating wrt this will give this filter's reaction to the image
			filter_specific_cost = output_volume[:,filter_i].sum()

			#===============================================================================
			# make the theano symbolic variable and then make a function of it, and then 
			# apply it to some input images 
			filter_saliency = theano.grad(filter_specific_cost, wrt=input_var)
			filter_saliency_fn = theano.function([input_var], filter_saliency)
			filter_saliency_images = filter_saliency_fn(X)

			filter_folder_name = os.path.join(layer_folder_name, "filter=%s"%filter_i)
			print "\t\tvisualizing filter %s"%filter_i

			with open(filter_folder_name, "w") as outfile:
				np.save(outfile, filter_saliency_images)

#===============================================================================
# Sample script
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()


saved_model = "../saved_models/mnist_c3_c3_c1_fc+addition-loop_Feb-27-2016_epoch=19"
# saved_model = "../saved_models/layers=5_loops=1_architecture-ID=10a222a5f3757ea7f2fa6cfafd3a514cdd22d8ca_Feb-20-2016_epoch=25"
model = LoopyNetwork(architecture_fpath="../architectures/mnist_c3_c3_c1_fc+loop.py", n_unrolls=2)

model.load_model(saved_model)



#===============================================================================
# below: modify the network so the gradients prooagate only positively, as 
# necessited by guided backprop.  This is necessary
relu = lasagne.nonlinearities.rectify
relu_layers = [layer for layer in lasagne.layers.get_all_layers(model.network)
               if getattr(layer, 'nonlinearity', None) is relu]
modded_relu = GuidedBackprop(relu)  # important: only instantiate this once!
for layer in relu_layers:
    layer.nonlinearity = modded_relu

#===============================================================================
# Now actually visualize
print X_train[0:2].shape
save_filter_visualizations_to_folder(model, dirname="../pictures", X=X_train[0:36], layers_to_vis=None, 
														filters_to_vis={}, 
														run_stem="test_gbp")



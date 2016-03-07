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


def save_filter_visualizations_to_folder(model, dirname, X, 
														layers_to_vis=None, 
														filters_to_vis={}, 
														run_stem="guided_backprop",
														positivize_gradients=True,
														dataset_name="cifar10"):
	"""
	Uses guided backprop to visualize all filters in all relevant layers.  Makes a folder containing all these, and within 
	that folder makes a folder for each layer, and within each one of those a folder for each filter, and with those reside
	the images.

	:param list layers_to_vis: a list of layer names to visualize.  If it is None, all layers are visualized.
	:param list filters_to_vis: a dict mapping layers in layers_to_vis to indexes of filters to visualize.  
			If there is no entry for an entry in layers_to_vis, then all filters for that layer are visualized.
	:param string run_stem: the stem of the name of the folder for this run
	:param string dirname: the name of the directory in which to save this folder
	:param array-like X: images with respect to which to get the guided bp.  Shape N, C, H, W I think
	:param LoopyNetwork model: a trained lasagne model.
	:param bool positivize_gradients: a boolean being True if we want to destructively modify the model in oder to 
		positivize gradients, as required by guided backprop
	"""
	if positivize_gradients:
		positivize_relu_gradients(model)

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

	if layers_to_vis is None:
		layers_to_vis = names_to_layers.keys()
	print "layers_to_vis: ", layers_to_vis
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
			filters_to_vis_for_this_layer = range(12)


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
			os.mkdir(filter_folder_name)
			print "\t\tvisualizing filter %s"%filter_i
			for image_i, image in enumerate(filter_saliency_images):
				saveto = os.path.join(filter_folder_name, "img=%s"%image_i)
				visualize_image(image, dataset=dataset_name, saveto=saveto)
			# with open(filter_folder_name, "w") as outfile:
			# 	np.save(outfile, filter_saliency_images)

if __name__=="__main__":
	#===============================================================================
	# Sample script
	# X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

	# saved_model = "../saved_models/mnist_c3_c3_c1_fc+addition-loop_Feb-27-2016_epoch=19"
	# saved_model = "../saved_models/layers=5_loops=1_architecture-ID=10a222a5f3757ea7f2fa6cfafd3a514cdd22d8ca_Feb-20-2016_epoch=25"
	# arch_fpath = "../architectures/mnist_c3_c3_c1_fc+loop.py"
	X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
	arch_fpath = "../architectures/cifar_isaac.py"
	saved_model = "../saved_models/cifar_c3-32_c3-64_c3-64_c3-3_fc_sumloop_Mar--5-15:38:29-2016_epoch=6"
	model = LoopyNetwork(architecture_fpath=arch_fpath, n_unrolls=2)
	model.load_model(saved_model)

	#===============================================================================
	# Now actually visualize
	save_filter_visualizations_to_folder(model, dirname="../pictures", 
												X=X_train[0:1], 
												layers_to_vis=None, 
												filters_to_vis={}, 
												run_stem="test_gbp",
												positivize_gradients=True)



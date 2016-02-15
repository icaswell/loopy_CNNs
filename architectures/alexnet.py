# File: alexnet.py
# @author: Lisa Wang
# @created: Feb 14 2016
# 

# Questions: What's the diff between template and layer_defaults?

{
"framework":"lasagne",

"templates": 
	{
	"input": {"type": "input"},
	"conv": {"type": "conv2d", "W": "glorot_uniform"},
	"dense": {"type": "dense", "W": "glorot_uniform"},
	},

#===============================================================================
# Here's where you define your layers.  You'll wire them together later.

"layers": #parser asserts that there exists a layer called "input" and a layer called "output"
	{
	"input":{"output_dim":(3,227, 227), "template": "input"},
	"conv_layer_1":{"num_filters":96, "template": "conv", "filter_size":11, "stride":4, "pad":0},  # TODO: how do I specify size and stride of filters?
	# "local_response_normalization_layer":{}, 
	# more info on local response normalization on slide 17 of 
	# http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf
	"max_pool_layer":{"filter_size":2, "stride":2}, # used after each of the first two conv layers
	
	"conv_layer_2":{"num_filters":256, "template": "conv", "filter_size":5, "stride":4, "pad":0},
	"conv_layer_3":{"num_filters":384, "template": "conv"},
	"conv_layer_4":{"num_filters":384, "template": "conv"},
	"conv_layer_5":{"num_filters":256, "template": "conv"},

	"dense_hidden_layer1":{"output_dim":4096, "template": "dense"},
	"dense_hidden_layer2":{"output_dim":4096, "template": "dense"},
	"top_layer":{"output_dim":1000, "template": "dense", "nonlinearity":"softmax"},
	},

"stacks":
	{
	"main_stack": {
		"type": "main",
		"structure": [
			"input", 
			"conv_layer_1", 
			# "local_response_normalization_layer"
			"max_pool_layer", 
			"conv_layer_2", 
			# "local_response_normalization_layer"
			"max_pool_layer", 
			"conv_layer_3", 
			"conv_layer_4", 
			"conv_layer_5", 
			"dense_hidden_layer1", 
			"dense_hidden_layer2", 
			"top_layer"]
		},			
	},

"layer_defaults":
	{
	"dense": 
		{		
		"W": "glorot_uniform",
		"nonlinearity": "relu"
		},	
	"conv2d": 
		{		
		"W": "glorot_uniform",
		"num_filters": 3,
		"filter_size": 3,
		"pad": 1,
		"nonlinearity": "relu"
		},			
	}
}


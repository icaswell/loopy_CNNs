# File: toy_loopy_mlp_lasagne_config.py
# @author: Isaac Caswell
# @created: Feb 7 2016
# 

{
"framework":"lasagne",

"templates": 
	{
	"input": {"type": "input"},
	"conv": {"type": "conv2d", "W": "glorot_uniform"},
	"pool": {"type": "pool2d", "pool_size":2},
	"dense": {"type": "dense", "W": "glorot_uniform"},
	},

#===============================================================================
# Here's where you define your layers.  You'll wire them together later.

"layers": #parser asserts that there exists a layer called "input" and a layer called "output"
	{
	"input":{"output_dim":(3,5,5), "template": "input"},
	"layer_1":{"num_filters":4, "template": "conv"},
	"layer_2":{"num_filters":7, "template": "conv"},
	"layer_2_pool":{"template": "pool"},
	"layer_3":{"num_filters":6, "template": "conv"},
	
	"loop_layer_1":{"num_filters":3, "template": "conv"},

	# "loop_layer_2":{"output_dim":14, "template": "conv"},
	# "loop_layer_3":{"output_dim":11, "template": "conv"},	
	"top_layer":{"output_dim":2, "template": "dense", "nonlinearity":"softmax"},
	},

"stacks":
	{
	"main_stack": {
		"type": "main",
		"structure": ["input", "layer_1", "layer_2", "layer_3", "layer_2_pool", "top_layer"]
		},
	"loop-1": {
		"type": "loop",
		"structure": ["layer_3", "loop_layer_1", "layer_1"],
		"composition_mode": 'mul'
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

		# class lasagne.layers.MaxPool2DLayer(incoming, pool_size, stride=None, pad=(0, 0), ignore_border=True, **kwargs)[source]	
	"pool2d":
		{
		}
	}
}


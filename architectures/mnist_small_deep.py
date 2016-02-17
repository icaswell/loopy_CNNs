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
	"input":{"output_dim":(1,28,28), "template": "input"},
	"conv_1":{"num_filters":6, "template": "conv"},
	"conv_2":{"num_filters":6, "template": "conv"},
	"conv_3":{"num_filters":6, "template": "conv"},
	"conv_4":{"num_filters":6, "template": "conv"},	
	"fc_1":{"output_dim":200, "template": "dense"},
	# "layer_2":{"num_filters":8, "template": "conv"},            
	# "layer_2_pool":{"template": "pool"},
	"fc_2":{"output_dim":10, "template": "dense", "nonlinearity":"softmax"},
	},

"stacks":
	{
	"main_stack": {
		"type": "main",
		"structure": ["input", "conv_1", "conv_2", "conv_3", "conv_4", "fc_1", "fc_2"]
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


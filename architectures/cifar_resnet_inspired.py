# File: toy_loopy_mlp_lasagne_config.py
# @author: Lisa Wang
# @created: Mar 4 2016
# Inspired by ResNet  (from He et al. : Deep Residual Learning for Image Recognition)

{
"run_params":
	{"use_batchnorm":True},
"framework":"lasagne",
"name": "cifar_c5-64_pool2_c3-64_c3-64_c3-64_c3-64_c3-64_c3-64_fc",

"templates": 
	{
	"input": {"type": "input"},
	"conv3": {"type": "conv2d", "W": "glorot_uniform", "filter_size": 3, "pad": 1},
	"conv5": {"type": "conv2d", "W": "glorot_uniform", "filter_size": 5, "pad": 2},	
	"pool": {"type": "pool2d", "pool_size":2},
	"dense": {"type": "dense", "W": "glorot_uniform"},
	},

#===============================================================================
# Here's where you define your layers.  You'll wire them together later.

"layers": #parser asserts that there exists a layer called "input" and a layer called "output"
	{
	"input":{"output_dim":(3,32,32), "template": "input"},
	"conv_1":{"num_filters":64, "template": "conv5"},
	"conv_2":{"num_filters":64, "template": "conv3"},
	"conv_3":{"num_filters":64, "template": "conv3"},
	"conv_4":{"num_filters":64, "template": "conv3"},
	"conv_5":{"num_filters":64, "template": "conv3"},
	"conv_6":{"num_filters":64, "template": "conv3"},
	"conv_7":{"num_filters":64, "template": "conv3"},
	"pool_l":{"template": "pool"},
	"loop_layer_1":{"num_filters":64, "template": "conv3"},
	"loop_layer_2":{"num_filters":64, "template": "conv3"}, 
	"fc_1":{"output_dim":10, "template": "dense", "nonlinearity":"softmax"},
	},

"stacks":
	{
	"main_stack": {
		"type": "main",
		"structure": ["input", "conv_1", "pool_l", "conv_2", "conv_3", "conv_4", "conv_5", "conv_6", "conv_7", "fc_1"]
		},
	"loop-1": {
		"type": "loop",
		"structure": ["conv_4", "loop_layer_1", "conv_2"],
		"composition_mode": "sum"
		},	
	"loop-2": {
		"type": "loop",
		"structure": ["conv_7", "loop_layer_2", "conv_5"],
		"composition_mode": "sum"
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


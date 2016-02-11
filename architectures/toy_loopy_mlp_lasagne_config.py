# File: toy_loopy_mlp_lasagne_config.py
# @author: Isaac Caswell
# @created: Feb 7 2016
# 

{
"framework":"lasagne",

"templates": 
	{
	"input": {"type": "input"},
	"dense": {"type": "dense", "W": "glorot_uniform"},
	},

#===============================================================================
# Here's where you define your layers.  You'll wire them together later.

"layers": #parser asserts that there exists a layer called "input" and a layer called "output"
	{
	"input":{"output_dim":(5,), "template": "input"},
	"layer_1":{"output_dim":11, "template": "dense"},
	"layer_2":{"output_dim":12, "template": "dense"},
	"layer_3":{"output_dim":13, "template": "dense"},
	
	# "loop_layer_1":{"output_dim":11, "template": "dense"},

	# "loop_layer_2":{"output_dim":11, "template": "dense"},
	# "loop_layer_3":{"output_dim":5, "template": "dense"},	
	"top_layer":{"output_dim":2, "template": "dense", "nonlinearity":"softmax"},
	},

"stacks":
	{
	"main_stack": {
		"type": "main",
		"structure": ["input", "layer_1", "layer_2", "layer_3", "top_layer"]
		},
	# "loop-1": {
	# 	"type": "loop",
	# 	"structure": ["layer_3", "loop_layer_1", "layer_2"],
	# 	"composition_mode": 'mul'
	# 	},
	# "loop-2": {
	# 	"type": "loop",
	# 	"structure": ["layer_2", "loop_layer_2", "loop_layer_3", "layer_1"],
	# 	"composition_mode": 'mul'
	# 	},				
	},

"layer_defaults":
	{
	"dense": 
		{		
		"W": "glorot_uniform",
		"nonlinearity": "relu"
		},	
	}
}


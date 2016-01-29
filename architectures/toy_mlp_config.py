# File: toy_mlp_config.py
# @author: Isaac Caswell
# @created: Jan 29 2016
# 

{
"templates": 
	{
	"input": {"type": "input"},
	"dense": {"type": "dense", "init": "Xavier_by_2"},
	},


#===============================================================================
# Here's where you define your layers.  You'll wire them together later.

"layers": #parser asserts that there exists a layer called "input" and a layer called "output"
	{
	"input":{"output_dim":374, "template": "input"},
	"layer_1":{"output_dim":500, "template": "dense"},
	"layer_2":{"output_dim":500, "template": "dense"}, 
	"output":{"output_dim":10, "template": "dense"},
	},

"stacks":
	{"main_stack": {
		"type": "main",
		"structure": ["input", "layer_1", "layer_2", "output"]
		},
	}

"layer_defaults":
	{
	"dense": 
		{
		"init": 'glorot_uniform',
		"activation": 'relu',
		"weights": None,
		"W_regularizer": None,
		"b_regularizer": None,
		"activity_regularizer": None,
		"W_constraint": None,
		"b_constraint": None,
		"input_dim": None,
		},
	},	
}


# File: architecture_config_readme.py
# @author: Isaac Caswell
# @created: Jan 28 2016
# 
# demonstrates how to specify the architecture of a (loopy) neural network using this
# config format.  Builds a toy loopy CNN andannotates what the different fields mean.
#
# USAGE:
#  from architecture_config_asserter import sanity_check
#  d = open("architectures/architecture_config_readme.py", "r").read()
#  architecture = eval(d)
#  sanity_check(architecture)

#
{
"framework":"lasagne",
#===============================================================================
# templates: this is where you define the different types of layers your network 
# will use.  Each template is required to have a "type" field.  The options may be 
# overridden by identically named options in the layer object.
# field.  The optional "descriptor" field provides a nice description of the node.
# The template "input" comes built-in.
#-------------------------------------------------------------------------------
# the "type" field may be one of:
#  	"conv": a conv layer
#	"dense": a fully connected layer
#-------------------------------------------------------------------------------
# note that these are merged into the defaults (defined at the end of this file)
"templates": #parser asserts that each has a type
	{
	"input": {"type": "input"},
	"conv_1": {"type": "conv2d",
				"stride": 1,
				"width": 5,
				"init": "glorot_uniform",}, #weight init
	"dense": {"type": "dense", 
				    	"init": "glorot_uniform"},
	# "loop_composition": {"type": "composition_mode",
	# 					"descriptor": "elementwise_multiplication",
	# 					"function": lambda x, y: x*y} #parser asserts that composition_function field exists
	},


#===============================================================================
# Here's where you define your layers.  You'll wire them together later.

"layers": #parser asserts that there exists a layer called "input" and a layer called "output"
	{
	"input":{"output_dim":374, "template": "input"},
	"layer_1":{"output_dim":500, "template": "conv_1"},
	"layer_2":{"output_dim":500, "template": "conv_1"}, 
	"layer_3":{"output_dim":10, "template": "dense"},
	"layer_4":{"output_dim":10, "template": "dense"},	
	#------------------------------------------------------------------------------
	"loop_1.1":{"output_dim":500, "template": "conv_1"},

	#------------------------------------------------------------------------------
	#"init" is being overridden from the "conv_1" template (which in turn comes from the "conv" default!	
	"loop_2.1":{"output_dim":500, "template": "conv_1", "init": "zero"},	
	"loop_2.2":{"output_dim":500, "template": "conv_1", "init": "zero"},
	},
# defines which layers are to be used as outputs, and what cost is to be used with them.
"outputs": {"layer_3": "mse"},
#===============================================================================
# the type must be one of the following:
# 	"main" - a normal bit of architecture
#	"loop" - a loopy connection. Note that these are handled quite differently!

"stacks": #parser asserts that there exists at least one layer with "input" at the beginning of it,
		  #and that there exists exactly one layer with output at the end of it.
	{"main_stack": {
		"type": "main",
		"structure": ["input", "layer_1", "layer_4", "layer_2", "layer_3"]
		},
	
	# {"residual":
	# 	"type": "main",
	# 	"structure": []
	# }
	"loop_1":{
		"type": "loop", 
		# parser asserts that all loop layers have a composition_mode
		"structure": ["layer_2", "loop_1.1", "layer_1"],
		"composition_mode": "mul"
		},
	"loop_2":{
		"type": "loop", 
		# parser asserts that all loop layers have a composition_mode
		"structure": ["layer_2", "loop_2.1", "loop_2.2",  "layer_1"],
		"composition_mode": "mul"
		},
		
	},

#===============================================================================
# the default parameters for layers.  Naming corresponds to keras conventions.
#
# activations: softplus, relu, tanh, sigmoid, hard_sigmoid, linear
# init: uniform, lecun_uniform, normal, identity, orthogonal, zero, glorot_normal, glorot_uniform, he_normal, he_uniform, 
"layer_defaults":
	{
	"dense": 
		{
		"init":'glorot_uniform',
		"activation":'relu',
		"weights":None,
		"W_regularizer":None,
		"b_regularizer":None,
		"activity_regularizer":None,
		"W_constraint":None,
		"b_constraint":None,
		# "input_dim":None,
		},
	"conv2d": # keras.layers.convolutional.Convolution2D(nb_filter,
		{
		 "init":'glorot_uniform',
		 "activation":'relu',
		 "weights":None,
		 "border_mode":'valid',
		 "subsample":(1, 1),
		 "dim_ordering":'th',
		 "W_regularizer":None,
		 "b_regularizer":None,
		 "activity_regularizer":None,
		 "W_constraint":None,
		 "b_constraint":None,
	 },	
	},	
}


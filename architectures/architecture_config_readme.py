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
#===============================================================================
# templates: this is where you define the different types of layers your network 
# will use.  Each template is required to have a "type" field.  The options may be 
# overridden by identically named options in the layer object.
# field.  The optional "descriptor" field provides a nice description of the node.
# The template "input" comes built-in.
#-------------------------------------------------------------------------------
# the "type" field may be one of:
#  	"conv": a conv layer
#	"fully_connected": a fully connected layer
#-------------------------------------------------------------------------------

"templates": #parser asserts that each has a type
	{
	"input": {"type": "input"},
	"conv_1": {"type": "conv",
				"stride": 1,
				"width": 5,
				"weight_init": "Xavier_by_2",},
	"fully_connected": {"type": "fully_connected", 
				    	"weight_init": "Xavier_by_2"},
	# "loop_composition": {"type": "composition_node",
	# 					"descriptor": "elementwise_multiplication",
	# 					"function": lambda x, y: x*y} #parser asserts that composition_function field exists
	},
#===============================================================================
# Here's where you define your layers.  You'll wire them together later.

"layers": #parser asserts that there exists a layer called "input" and a layer called "output"
	{
	"input":{"dim":374, "template": "input"},
	"layer_1":{"dim":500, "template": "conv_1"},
	"layer_2":{"dim":500, "template": "conv_1"}, 
	"output":{"dim":10, "template": "fully_connected"},
	#------------------------------------------------------------------------------
	"loop_1.1":{"dim":500, "template": "conv_1"},

	#------------------------------------------------------------------------------
	#"width" is being overridden from the "conv_1" template!	
	"loop_2.1":{"dim":500, "template": "conv_1", "width": 3},	
	"loop_2.2":{"dim":500, "template": "conv_1", "width": 3},
	},

#===============================================================================
# the type must be one of the following:
# 	"main" - a normal bit of architecture
#	"loop" - a loopy connection. Note that these are handled quite differently!

"stacks": #parser asserts that there exists at least one layer with "input" at the beginning of it,
		  #and that there exists exactly one layer with output at the end of it.
	{"main_stack": {
		"type": "main",
		"structure": ["input", "layer_1", "layer_2", "output"]
		},
	
	# {"residual":
	# 	"type": "main",
	# 	"structure": []
	# }
	"loop_1":{
		"type": "loop", 
		# parser asserts that all loop layers have a composition_node
		"structure": ["layer_2", "loop_1.1", "layer_1"],
		"composition_node": "loop_composition"
		},
	"loop_2":{
		"type": "loop", 
		# parser asserts that all loop layers have a composition_node
		"structure": ["layer_2", "loop_2.1", "loop_2.2",  "layer_1"],
		"composition_node": lambda x, y: x*y
		},
		
	}
}


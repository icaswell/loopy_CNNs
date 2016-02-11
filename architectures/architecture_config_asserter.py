# File: architecture_config_asserter.py
# @author: Isaac Caswell
# @created: Jan 28 2016
# 
# performs various assertions on some architecture that one has specified in 
# architecture_config_readme.py.  These are for the purpose of sanity checking.
#
# TODO: better warnings; check that each layer has only one input
# TODO: assert that the ends of loops match their input layers' input dimensions


def sanity_check(architecture):
	assert "framework" in architecture
	assert architecture["framework"] in ["keras", "lasagne"]
	for t_name, template in architecture['templates'].items():
		assert "type" in template
		assert template["type"] in ["conv2d", "dense", "dropout", "input"]

	assert "input" in architecture["layers"]
	# assert "output" in architecture["layers"]
	
	stack_beginnings = []
	stack_ends = []
	all_used_layers = set()
	stack_types = []
	contains_main_stack = False
	for stack_name, stack in architecture["stacks"].items():
		assert "type" in stack
		assert  stack["type"] in ["main", "loop", "residual"]
		contains_main_stack  = contains_main_stack or stack["type"] == "main_stack"
		stack_types.append(stack["type"])
		stack_beginnings.append(stack["structure"][0])
		stack_ends.append(stack["structure"][-1])
		if stack["type"] == "loop":
			assert "composition_mode" in stack
		all_used_layers |= set(stack["structure"])

		assert all([layer_name in architecture["layers"] for layer_name in stack["structure"]])


	assert stack_types.count("main") > 0
	if stack_types.count("main") > 1:
		print "severe warning: use no more than one main stack until I get cleverer"
	# assert stack_ends.count("output") == 1
	assert stack_beginnings.count("input") >= 1


	for layer_name, layer in architecture["layers"].items():
		assert layer_name in all_used_layers

	#TODO: assert that the top layer has softmax nonlinearity on it, or at least give big warning
	print "Good job.  You have followed directions.  Asserter passes."

if __name__=="__main__":
	with open("architecture_config_readme.py", "r") as f:
		arch_string = f.read()
	arch_dict = eval(arch_string)
	sanity_check(arch_dict)

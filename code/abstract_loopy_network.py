# abstract_loopy_network.py
# @author: Isaac Caswell
# @created: Jan 28 2016
#
#===============================================================================
# DESCRIPTION:
# 
#  An abstract class that allows the structure of a lopy neural network to be 
# defined in a separate config file.  The point of this class is that it is 
# Framework-independent, so you can make the inheriting class in keras, lasagne, 
# raw tensorflow, etc.
#===============================================================================
# CURRENT STATUS: seems to work
#===============================================================================
# USAGE:
# 
# from abstract_loopy_network import AbstractLoopyNetwork
# class LoopyNetwork(AbstractLoopyNetwork):
#   ...override the relevant methods
#===============================================================================
# TODO:
# -doc more
# -expand parsing of config file uch that one can define multiple outputs (right 
# now we just take the top of the main_stack)



from collections import defaultdict
import sys
from pprint import pprint
import hashlib
sys.path.append("../architectures")


from architecture_config_asserter import sanity_check
import util

class AbstractLoopyNetwork():
    def __init__(self, architecture_fpath, n_unrolls, debug=True):
        """
        __init__ must be overridden.  
        """
        self.debug=debug
        self.debug_colors = ["red", "blue", "peach", "magenta_highlight"]
        self.n_unrolls = n_unrolls
        self.architecture_dict = self._prep_layer_dicts(architecture_fpath)     
        self._build_description(self.architecture_dict)



    def _prep_layer_dicts(self, architecture_fpath):
        """
        expands the options of each layer to include the defaults for its template,
        as well as the global defaults.  Stores these in the "options" field of the
        dict for that layer.
        """

        with open(architecture_fpath, "r") as f:
            string_representation = f.read()
        architecture_dict = eval(string_representation)
 
        # sanity check: make sure that the architecture is consistent
        # and well configured.
        sanity_check(architecture_dict)

        for layer_name, layer_dict in architecture_dict["layers"].items():
            template = architecture_dict["templates"][layer_dict["template"]]
            architecture_dict["layers"][layer_name]["type"] = template["type"]

            default = dict(architecture_dict["layer_defaults"][template["type"]]) if template["type"] in architecture_dict["layer_defaults"] else {}

            default.update(template)
            default.update(layer_dict)
            # default = {key:value for key, value in default.items() if key in self._POSSIBLE_KERAS_OPTIONS}
            for non_framework_option in ["template", "output_dim", "type"]:
                if non_framework_option in default:
                    del default[non_framework_option]
            architecture_dict["layers"][layer_name]["options"] = default

        return architecture_dict
 

    def _label(self, name, unroll_i):
        return name + "_unroll=%s"%unroll_i

    def _add_input(self, name, input_shape):
        """
        Adds an input layer.  Depending on the framework used, this could mean a variety of things.
        """
        print "Error: this method must be overriden by subclass"
        sys.exit(0)


    def _debug_print(self, message, color_id=0):
        console_identifier = "ALN> "
        if self.debug:
            util.colorprint(console_identifier, self.debug_colors[-1], newline=False)
            util.colorprint(message, self.debug_colors[color_id])
                  

    def _build_architecture(self, architecture_dict):
        """
        Builds the architecture of the neural net.  
        Assumptions:
            -there is oly one main_stack, which begins in the input and ends in the output.
        unrolls the nework self.n_unrolls times, duplicating the entire main_stack each time.
        """
        loop_outputs = {} # outputs from loops
        loops = [stack for stack_name, stack in architecture_dict['stacks'].items() if stack['type']=='loop']
        main_stack = architecture_dict['stacks']['main_stack']

        layers_with_loops = [loop_layer for loop in loops for loop_layer in loop['structure']]
        main_stack_layers_interacting_with_loops = [i for i, layer in enumerate(main_stack['structure']) if layer in layers_with_loops]
        if not main_stack_layers_interacting_with_loops:
            #there are no loops
            main_stack_layers_interacting_with_loops  = [len(main_stack['structure'])]

        last_layer_added="input"
        self._add_input(name=last_layer_added, input_shape=architecture_dict["layers"]["input"]["output_dim"])

        #===============================================================================
        # Below loop: build the architecture up until the loopy section starts
        for underlying_layer_name in main_stack['structure'][1:main_stack_layers_interacting_with_loops[0]]:
            layer_dict = architecture_dict['layers'][underlying_layer_name]
            self._debug_print("[pre-loop] adding layer %s with input %s"%(underlying_layer_name, last_layer_added), 1)

            last_layer_added = self._add_layer(layer_dict, 
                                            layer_name=underlying_layer_name,
                                            input_layers=last_layer_added
                                            )       
        #===============================================================================
        # Below loop: build the architecture in the section with loops in it n_unrolls times
        first_layer_in_loopy_section = last_layer_added
        for unroll_i in range(self.n_unrolls):
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # NOTE: this is an interesting structureal decision to add the input at the 
            # beginning of every unroll, and abstracts it from the resnetty feel.
            # TODO: I want to add a time decay factor here/an increasingly powerful dropout mask/something
            last_layer_added=first_layer_in_loopy_section
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Nomenclature note: 
            # 'underlying layer name' corresponds to the layer in the loopy graph, e.g. "layer_1"
            # 'last_layer_added' is the name of the layer in the unrolled graph, e.g. "layer_1_unroll=0"

            self._debug_print("loop outputs: %s"%", ".join(str(out) for out in loop_outputs.items()), 3)
            # print "loop_outputs: ", loop_outputs

            Lisa = main_stack_layers_interacting_with_loops[0]
            Isaac = main_stack_layers_interacting_with_loops[-1]


            for underlying_layer_name in main_stack['structure'][Lisa : Isaac+1]:
                layer_dict = architecture_dict['layers'][underlying_layer_name]

                merge_mode = None # this will be set to non-none iff there is an incoming loop
                if underlying_layer_name in loop_outputs:
                    # TODO: fix to list addition
                    last_layer_added = [last_layer_added, loop_outputs[underlying_layer_name][0]]
                    merge_mode = loop_outputs[underlying_layer_name][1]
                # util.colorprint("%s with input %s"%(self._label(underlying_layer_name, unroll_i), last_layer_added), 'blue')
                self._debug_print("[repeating section] adding layer %s with input %s"%(self._label(underlying_layer_name, unroll_i), last_layer_added), 0)

                last_layer_added = self._add_layer(layer_dict, 
                                                layer_name=self._label(underlying_layer_name, unroll_i),
                                                input_layers=last_layer_added, 
                                                merge_mode=merge_mode,
                                                share_params_with = self._label(underlying_layer_name, 0) if unroll_i else None
                                                )
            #============================================================================
            # Below clause: calculate the loop outputs 
            if unroll_i < self.n_unrolls - 1: #don't calculate outputs for the last unroll
                for loop_dict in loops:
                    loop_output_name = self.add_loop(loop_dict, unroll_i=unroll_i, architecture_dict=architecture_dict)
                    #TODO: fix! Edit: what needs fixing?
                    loop_outputs[loop_dict['structure'][-1]] = loop_output_name, loop_dict["composition_mode"]

        #===============================================================================
        # Below loop: finish the main stack after the loopy part

        for underlying_layer_name in main_stack['structure'][main_stack_layers_interacting_with_loops[-1]+1:]:
            self._debug_print("[after repeating section] adding layer %s with input %s"%(underlying_layer_name, last_layer_added), 1)

            layer_dict = architecture_dict['layers'][underlying_layer_name]
            
            last_layer_added = self._add_layer(layer_dict, 
                                            layer_name=underlying_layer_name,
                                            input_layers=last_layer_added
                                            )       
        #===============================================================================
        #                     
        #===============================================================================
        # Adds the last added layer as an output
        self._debug_print("marking layer %s as output"%last_layer_added, 3)
        self._add_output(name="output", input_layer=last_layer_added)


    def add_loop(self, loop_dict, unroll_i, architecture_dict):
        """
        adds a loop to the model, by linking the beginning of the 
        loop to the relevant layer from THIS unroll.

        Note that this does NOT connect the end of the loop--
        that's left hanging, for the calling function to deal with.
        """
        #get the input from the previous layer
        self._debug_print("adding loop:", 0)

        cur_layer = self._label(loop_dict["structure"][0], unroll_i)
        for loop_layer_name in loop_dict["structure"][1:-1]: #don't include the input or the output
            # print "adding %s to the loop"%loop_layer_name
            share_params_with = self._label(loop_layer_name, 0) if unroll_i else None
            self._debug_print("[in loop] adding layer %s with input %s, sharing parameters with %s"%(loop_layer_name, cur_layer, share_params_with), 3)
            pprint(architecture_dict["layers"][loop_layer_name])
            cur_layer = self._add_layer(layer_dict=architecture_dict["layers"][loop_layer_name], 
                                        layer_name=self._label(loop_layer_name, unroll_i),
                                        input_layers=cur_layer,
                                        share_params_with = share_params_with
                                        )
        return cur_layer


    def _add_layer(self, layer_dict, layer_name, id, input_layers, share_params_with=None, merge_mode=None):
        """
        input_layers may be either a string or a list.  If it's a list (meaning that there's 
            some loop input), all incoming acivations are merged via merge_mode.
        """
        print "Error: this method must be overriden by subclass"
        sys.exit(0)    

    def _build_description(self, architecture_dict):
        """
        :param dict architecture_dict: a dict from the config file, that 
            represents the architecture of the model
        makes a pretty colored string that represents the architecture of the 
        model.  Does not yet represent the loops, which is a pity.  But how is one 
        to do that graphically on terminal?
        """
        #TODO: once one handles outputs properly, color them differently.

        title_color = 95
        layer_colors = {
                "input": 93,
                "conv2d": 96,
                "dense": 97,
                "output": 98,  
                "pool2d": 92, 
        }
        self.description = "LoopyCNN instance with the following hyperparameters, layers and loops:"
        self.description += "\033[%sm"%title_color + "\nHYPERPARAMETERS:" + '\033[0m'
        self.description += "\n\tn_unrolls=%s"%self.n_unrolls
        self.description += "\n\tuse_batchnorm=%s"%architecture_dict.get("run_params", {}).get("use_batchnorm", False)
        self.description += "\033[%sm"%title_color + "\n\nARCHITECTURE:" + '\033[0m'
        for stack_name, stack in architecture_dict["stacks"].items():
            self.description +="\n%s:"%stack_name
            for layer_name in stack["structure"]:
                layer = architecture_dict["layers"][layer_name]
                if "template" in layer:
                    template = architecture_dict["templates"][layer["template"]]
                    template.update(layer)
                    layer=template
                # TODO: replace this with looping through the actual layers, each of which will have a __str__() method
                layer_desc = layer_name
                if layer["type"] == "conv2d":
                    layer_desc += " [%s layer; num_filters=%s]"%(layer["type"], layer["num_filters"])
                elif layer["type"] == "pool2d":
                    layer_desc += " [pooling layer]" 
                else:
                    layer_desc += " [%s layer; output_dim=%s]"%(layer["type"], layer["output_dim"])                    
                layer_color = layer_colors[layer["type"]]
                self.description += "\n\t\033[%sm"%layer_color + layer_desc + '\033[0m'

        #===============================================================================
        # get hash id and short descriptor
        self.get_hash_id()
        n_loops = sum(1 for stack in architecture_dict["stacks"].values() if stack["type"] =="loop")
        n_layers_in_main_stack = len(architecture_dict["stacks"]["main_stack"]["structure"]) -1
        self.short_description = "\033[104m %s-layer network with %s loops (architecture ID: %s)\033[0m"%(n_layers_in_main_stack, n_loops, self._hash_id)
        self.architecture_name = architecture_dict.get("name", "layers=%s_loops=%s_architecture-ID=%s"%(n_layers_in_main_stack, n_loops, self._hash_id))



    def get_hash_id(self):
        """
        makes a unique hash for a model.  This is useful if you want to know whether two 
        models are identical or not.
        """
        self._hash_id = hashlib.sha1(self.description).hexdigest()

    def __repr__(self):
        """
        returns a descriptive string representation of the model.
        """
        return self.description

    def __str__(self):
        """
        returns a short string representation of the model.
        """
        return self.short_description     

if __name__=="__main__":
     model = AbstractLoopyNetwork(architecture_fpath="../architectures/toy_mlp_config.py", n_unrolls=3)

     print repr(model)

     print "The model is called %s"%model


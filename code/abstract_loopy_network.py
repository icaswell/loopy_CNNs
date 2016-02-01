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
sys.path.append("../architectures")


from architecture_config_asserter import sanity_check

class AbstractLoopyNetwork():
    def __init__(self, architecture_fpath, n_unrolls):
        """
        __init__ must be overridden.  
        """
        self.n_unrolls = n_unrolls
        architecture_dict = self._prep_layer_dicts(architecture_fpath)     
        self._build_description(architecture_dict)



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
 
    def __repr__(self):
        """
        returns a descriptive string representation of the model.
        """
        return self.description

    def _label(self, name, unroll_i):
        return name + "_unroll=%s"%unroll_i

    def _add_input(self, name, input_shape):
        """
        Adds an input layer.  Depending on the framework used, this could mean a variety of things.
        """
        print "Error: this method must be overriden by subclass"
        sys.exit(0)


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

        cur_layer_name="input"
        #TODO: don't hardcode input value
        self._add_input(name=cur_layer_name, input_shape=architecture_dict["layers"]["input"]["output_dim"])
        # TODO: add input
        for unroll_i in range(self.n_unrolls):
            for layer_name in main_stack['structure'][1:]:
                layer_dict = architecture_dict['layers'][layer_name]


                merge_mode = None # this will be set to non-none iff there is an incoming loop
                if layer_name in loop_outputs:
                    cur_layer_name = [cur_layer_name, loop_outputs[layer_name][0]]
                    merge_mode = loop_outputs[layer_name][1]

                cur_layer_name = self._add_layer(layer_dict, 
                                                layer_name=self._label(layer_name, unroll_i),
                                                input_layers=cur_layer_name, 
                                                merge_mode=merge_mode,
                                                share_params_with = self._label(layer_name, 0) if unroll_i else None
                                                )
            #====================================================
            # calculate the loop outputs 
            if unroll_i < self.n_unrolls - 1:
                #don't calculate outputs for the last unroll
                for loop_dict in loops:
                    loop_output_name = self.add_loop(loop_dict, unroll_i=unroll_i, architecture_dict=architecture_dict)
                    loop_outputs[loop['structure'][-1]] = loop_output_name, loop_dict["mode"]
        #===============================================================================
        # Adds the last added layer as an output
        self._add_output(name="output", input_layer=cur_layer_name)


    def add_loop(self, loop_dict, unroll_i, architecture_dict):
        """
        adds a loop to the model, by linking the beginning of the 
        loop to the relevant layer from the last unroll.

        Note that this does NOT connect the end of the loop--
        that's left hanging, for the callign function to deal with.
        """
        #get the input from the previous layer
        cur_layer = self._label(loop_dict["structure"][0], unroll_i - 1.0)
        for loop_layer_name in loop_dict["structure"][1:-1]: #don't include the input or the output
            cur_layer = self._add_layer(architecture_dict["layers"][loop_layer_name], 
                                        layer_name=self._label(loop_layer_name, unroll_i),
                                        input_layer=cur_layer,
                                        share_params_with = self._label(loop_layer_name, 0) if unroll_i else None
                                        )
        return cur_layer

    def _label(self, name, unroll_i):
        return name + "_unroll=%s"%unroll_i

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
        model.  Does not yer represent the loops, which is a pity.  But how is one 
        to do that graphically on terminal?
        """
        #TODO: once one handles outputs properly, color them differently.
        title_color = 95
        layer_colors = {
                "input": 93,
                "conv2d": 96,
                "dense": 97,
                "output": 98,   
        }
        self.description = "LoopyCNN instance with the following hyperparameters, layers and loops:"
        self.description += "\033[%sm"%title_color + "\nHYPERPARAMETERS:" + '\033[0m'
        self.description += "\n\tn_unrolls=%s"%self.n_unrolls
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
                layer_desc += " [%s layer; output_dim=%s]"%(layer["type"], layer["output_dim"])
                layer_color = layer_colors[layer["type"]]
                self.description += "\n\t\033[%sm"%layer_color + layer_desc + '\033[0m'


    def __repr__(self):
        """
        returns a descriptive string representation of the model.
        """
        return self.description


if __name__=="__main__":
     model = AbstractLoopyNetwork(architecture_fpath="../architectures/toy_mlp_config.py", n_unrolls=2)

     print repr(model)


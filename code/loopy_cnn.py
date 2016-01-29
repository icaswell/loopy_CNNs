# loopy_cnn.py
# @author: Isaac Caswell
# @created: Jan 28 2016
#
#===============================================================================
# DESCRIPTION:
#
# Defines the LoopyCNN class.  It is designed to build architecture that's defined
# in a config file.  (see architecture_config_readme.py for an example)
#
#===============================================================================
# CURRENT STATUS: Massively Unimplemented
#===============================================================================
# USAGE:
# from loopy_cnn import LoopyCNN
# model = LoopyCNN(architecture_fpath="../architectures/simple_loop.py", 
#         **kwargs)

# keras.layers.core.Dense(output_dim,
# init='glorot_uniform',
# activation='linear',
# weights=None,
# W_regularizer=None,
# b_regularizer=None,
# activity_regularizer=None,
# W_constraint=None,
# b_constraint=None,
# input_dim=None)

# keras.layers.convolutional.Convolution2D(nb_filter,
#  nb_row,
#  nb_col,
#  init='glorot_uniform',
#  activation='linear',
#  weights=None,
#  border_mode='valid',
#  subsample=(1, 1),
#  dim_ordering='th',
#  W_regularizer=None,
#  b_regularizer=None,
#  activity_regularizer=None,
#  W_constraint=None,
#  b_constraint=None)


from collections import defaultdict
import sys
sys.path.append("../architectures")

import keras
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
 

from architecture_config_asserter import sanity_check


class LoopyCNN():
    def __init__(self, architecture_fpath, 
                    n_unrolls=2, 
                    optimizer = "rmsprop"):

        self._POSSIBLE_KERAS_OPTIONS = set(["init", "activation", "weights", "W_regularizer", "b_regularizer", "activity_regularizer", "W_constraint", "b_constraint", "input_dim", "border_mode", "subsample", "dim_ordering", "activity_regularizer"])

        self.description = "uninitialized LoopyCNN instance"        
        self.n_unrolls = n_unrolls
        self._init_optimizer(optimizer)

        self._build_architecture(architecture_fpath)
        self.debug_model_color = "red"


 
    def train_model(self):
        pass
 
    def classify_batch(self):
        pass

    def _prep_layer_dicts(self, architecture):
        for layer_name, layer_dict in architecture["layers"].items():
            template = architecture["templates"][layer_dict["template"]]
            architecture["layers"][layer_name]["type"] = template["type"]

            default = dict(architecture["layer_defaults"][template["type"]]) if template["type"] in architecture["layer_defaults"] else {}
            default.update(template)
            default.update(layer_dict)
            default = {key:value for key, value in default.items() if key in self._POSSIBLE_KERAS_OPTIONS}
            architecture["layers"][layer_name]["keras_options"] = default
 
    def _build_architecture(self, architecture_fpath):
        with open(architecture_fpath, "r") as f:
            string_representation = f.read()
        architecture_dict = eval(string_representation)
 
        # sanity check: make sure that the architecture is consistent
        # and well configured.
        sanity_check(architecture_dict)
 
        self._prep_layer_dicts(architecture_dict)
        self._build_description(architecture_dict)


        #TODO: remove
        return
        self.model = Graph()

        loop_outputs = {} # outputs from loops
        loops = [stack for stack_name, stack in architecture_dict['stacks'].items() if stack['type']=='loop']
        main_stack = architecture_dict['stacks']['main_stack']

        cur_layer_name="input"
        #TODO: don't hardcode input value
        self.model.add_input(name=cur_layer_name, input_shape=(32,))
        # TODO: add input
        for unroll_i in range(self.n_unrolls):
            for layer_name in main_stack['structure'][1:]:
                layer_dict = architecture_dict['layers'][layer_name]


                merge_mode = None # this will be set to non-none iff there is an incoming loop
                if layer_name in loop_outputs:
                    cur_layer_name = [cur_layer_name, loop_outputs[layer_name][0]]
                    merge_mode = loop_outputs[layer_name][1]

                cur_layer_name = self._add_layer(layer_dict, name=layer_name, id=unroll_i, input_layers=cur_layer_name, merge_mode=merge_mode)
            # TODO: add output
                
            #====================================================
            # calculate the loop outputs 
            if unroll_i < self.n_unrolls - 1:
                #don't calculate outputs for the last unroll
                for loop_dict in loops:
                    loop_output_name = self._add_loop(loop_dict, unroll_i=unroll_i, architecture_dict=architecture_dict)
                    loop_outputs[loop['structure'][-1]] = loop_output_name, loop_dict["mode"]

    def _add_layer(self, layer_dict, name, id, input_layers, merge_mode=None):
        """
        input_layers may be either a string or a list.  If it's a list (meaning that there's 
            some loop input), all incoming acivations are merged via merge_mode.
        """
        
        layer_options = layer_dict["keras_options"]
        layer=None
        if layer_dict["type"]=="conv2d":
            #TODO: remove below
            nb_filter, nb_row, nb_col = 3,3,3
            layer = keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, **layer_options)
        elif layer_dict["type"]=="dense":
            layer = keras.layers.core.Dense(**layer_options)       
        else:
            print "ur sol"
            sys.exit(0)
        # TODO: one of the layers is a string
        if isinstance(input_layers, list):
            #this means that there is input from a loop to this layer
            merged_name = " * ".join(layers_to_merge) #TODO-this assumes all merges are multiplication            
            self.model.add_node(layer, name=layer_name, inputs=input_layers, merge_mode=merge_mode)
        else:
            layer_name = self._label(name, id)
            self.model.add_node(layer, name=layer_name, input=input_layers)

        return layer_name      


    def _add_loop(self, loop_dict, unroll_i, architecture_dict):
        #get the input from the previous layer
        cur_layer = self._label(loop_dict["structure"][0], unroll_i - 1.0)
        for loop_layer_name in loop_dict["structure"][1:-1]: #don't include the input or the output
            cur_layer = self._add_layer(architecture_dict["layers"][loop_layer_name], 
                                        name=loop_layer_name,
                                        id=unroll_i,
                                        input_layer=cur_layer)
        return cur_layer

    def _label(self, name, unroll_i):
        return name + "_unroll=%s"%unroll_i


    def _build_description(self, architecture_dict):
        title_color = 95
        layer_colors = {
                "input": 93,
                "conv2d": 96,
                "dense": 97,
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


    def _init_optimizer(self, optimizer):
        """
        converts a string to a keras optimizer
        """
        if optimizer == "rmsprop":
            self.optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        elif optimizer == "adagrad":
            self.optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
        elif optimizer == "adadelta":
            self.optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        elif optimizer == "adam":
            self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        elif optimizer == "adamax":
            self.optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)                                  
        elif hasattr(optimizer, __call__):
            self.optimizer = optimizer
        else:
            print "Error: unsupported optimizer %s"%optimizer
            sys.exit(0)
    def __repr__(self):
        """
        returns a descriptive string representation of the model.
        """
        return self.description


if __name__=="__main__":
     model = LoopyCNN(architecture_fpath="../architectures/architecture_config_readme.py")

     print repr(model)


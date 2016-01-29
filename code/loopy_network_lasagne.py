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


class LoopyNetwork():
    def __init__(self, architecture_fpath, 
                    n_unrolls=2, 
                    optimizer = "rmsprop"):

        self.n_unrolls = n_unrolls
        self._init_optimizer(optimizer)
        #self.prep_layer also sets slf.__repr__
        architecture_dict = self._prep_layer_dicts(architecture_fpath)     

        self._build_architecture(architecture_dict)
        self.debug_model_color = "red"


 
    def train_model(self):
        pass
 
    def classify_batch(self):
        pass

 
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



if __name__=="__main__":
     model = LoopyNetwork(architecture_fpath="../architectures/architecture_config_readme.py")

     print repr(model)


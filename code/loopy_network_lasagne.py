# loopy_cnn.py
# @author: Isaac Caswell
# @created: Jan 31 2016
#
#===============================================================================
# DESCRIPTION:
#
# Defines the LoopyCNN class, as implemented in lasagne. 
#
#===============================================================================
# CURRENT STATUS: Massively Unimplemented
#===============================================================================
# USAGE:
# from loopy_cnn import LoopyCNN
# model = LoopyCNN(architecture_fpath="../architectures/simple_loop.py", 
#         **kwargs)


import numpy as np
from collections import defaultdict
import sys
sys.path.append("../architectures")

import theano
import theano.tensor as T
import lasagne

from abstract_loopy_network import AbstractLoopyNetwork


class LoopyNetwork(AbstractLoopyNetwork):
    def __init__(self, architecture_fpath, 
                    n_unrolls=2, 
                    optimizer = "rmsprop",
                    loss="mse"):

        
        self.n_unrolls = n_unrolls        
        self.loss=loss
        self._init_optimizer(optimizer)
        self.outputs = {} #mapping from output layer name to the layer that it refers to.
        # self.model = Graph()

        #self.prep_layer also sets slf.__repr__
        architecture_dict = self._prep_layer_dicts(architecture_fpath)     
        self._build_architecture(architecture_dict)
        self._build_description(architecture_dict)

        # self.names_to_layers: a mapping of string (e.g. "layer_1_unroll=2") to a lasagne layer object
        self._names_to_layers = {}


        
    def train_model(self, X_train, y_train, n_epochs=10):
        self.model.compile(optimizer=self.optimizer, loss={self.output :self.loss})
        return self.model.fit({'input':X_train, self.output:y_train}, nb_epoch=n_epochs)
 
    def classify_batch(self):
        pass

    #===============================================================================
    # private functions

    def _add_input(self, name, input_shape):
        input_layer = lasagne.layers.InputLayer(shape=input_shape,
                                    input_var=None)
        self._names_to_layers[name] = input_layer

    def _add_output(self, name, input_layer):
        """
        input_layer is a string, e.g. "dense1"
        """
        self.outputs[name] = self._names_to_layers[input_layer]

    def _convert_layer_options(layer_options):
        """
        takes a mapping of 
            parameter name: description of that parameter
        to
            parameter name: lasagne object corresponding thereto

        also rectifies the naming scheme with lasagne rather than keras names.
        e.g.
            "acivation": relu
        to
            "nonlinearity": lasagne.nonlinearities.rectify
        """
        parameter_name_mapping = {
                "activation": "nonlinearity"
        }

        for keras_name, lasagne_name in parameter_name_mapping.items():
            if keras_name in layer_options:
                layer_options[lasagne_name] = layer_options[keras_name]
                del layer_options[keras_name]

        if "nonlinearity" in layer_options:
            nonlinearity = {"relu": lasagne.nonlinearities.rectify,
            }[layer_options["nonlinearity"]]
            layer_options["nonlinearity"] = nonlinearity

        #TODO: everything.


    def _add_layer(self, layer_dict, layer_name, input_layers, merge_mode=None, share_params_with=None):
        """
        input_layers may be either a string or a list.  If it's a list (meaning that there's 
            some loop input), all incoming acivations are merged via merge_mode.
        """
        layer_dict = dict(layer_dict)
        assert isinstance(input_layers, str) #TODO: remove after figuring out layers in lasagne
        
        layer_options = layer_dict["lasagne_options"]
        layer_options = self._convert_layer_options(layer_options)
        layer=None

        if share_params_with is not None:
            layer_options["W"] = ....

        if layer_dict["type"]=="conv2d":
            #TODO: remove below
            RaiseError()
            # nb_filter, nb_row, nb_col = 3,3,3
            # layer = keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, **layer_options)
        elif layer_dict["type"]=="dense":
            dim  = layer_dict["output_dim"]
            # del layer_options["output_dim"]

            layer = lasagne.layers.DenseLayer(
                        self._names_to_layers[input_layers], 
                        num_units=dim,
                        **layer_options, 
                        # nonlinearity=lasagne.nonlinearities.rectify,
                        # W=lasagne.init.GlorotUniform(),
                    )

        else:
            print "ur sol"
            RaiseError()
        # TODO: one of the layers is a string
        # if isinstance(input_layers, list):
        #     #this means that there is input from a loop to this layer
        #     self.model.add_node(layer, name=layer_name, inputs=input_layers, merge_mode=merge_mode)
        # else:
        #     self.model.add_node(layer, name=layer_name, input=input_layers)

        return layer_name      


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
    model = LoopyNetwork(architecture_fpath="../architectures/toy_mlp_config.py", n_unrolls=1)

    print repr(model)
    X_train = np.zeros((0,5))
    y_train = np.zeros((0,2))
    with open("../data/toy_data_5d.txt", "r") as f:
        for line in f:
            splitline=line.split()
            yi = int(splitline[-1])
            xi = [float(xij) for xij in splitline[0:-1]]
            d = len(xi)
            xi = np.array(xi)
            xi = xi.reshape((1, d))

            yi_expanded = np.zeros((1,2))
            yi_expanded[0,yi] = 1.0

            X_train = np.vstack([X_train, xi])
            y_train = np.vstack([y_train, yi_expanded])



    model.train_model(X_train, y_train, n_epochs=120)

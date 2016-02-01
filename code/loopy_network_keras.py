# loopy_cnn.py
# @author: Isaac Caswell
# @created: Jan 28 2016
#
#===============================================================================
# DESCRIPTION:
#
# Defines the LoopyCNN class, implemented in keras.  It is designed to build architecture that's defined
# in a config file.  (see architecture_config_readme.py for an example)
#
#===============================================================================
# CURRENT STATUS: Works, but doesn't have parameter sharing for loops (e.g. is basically a resnet)
# Also only works for dense layers right now.
#===============================================================================
# USAGE:
# from loopy_cnn import LoopyCNN
# model = LoopyCNN(architecture_fpath="../architectures/simple_loop.py", 
#         **kwargs)


import numpy as np
from collections import defaultdict
import sys
sys.path.append("../architectures")

import keras
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
# from keras.utils.dot_utils import Grapher
 

from abstract_loopy_network import AbstractLoopyNetwork
import util
import plot_keras as plotter


class LoopyNetwork(AbstractLoopyNetwork):
    def __init__(self, architecture_fpath, 
                    n_unrolls=2, 
                    optimizer = "rmsprop",
                    loss="mse"):
        #===============================================================================
        # Call the superclass init function.  The commented out line is for python 3.
        # super(LoopyNetwork, self).__init__(architecture_fpath, n_unrolls)
        AbstractLoopyNetwork.__init__(self, architecture_fpath, n_unrolls)

        print "WARNING: this does not actually create a loopy network (parameters are not shared)"
        self.short_name = "loopy_keras"        

        self.loss=loss
        self._init_optimizer(optimizer)
        self.model = Graph()
        self._POSSIBLE_KERAS_OPTIONS = set(["init", "activation", "weights", "W_regularizer", "b_regularizer", "activity_regularizer", "W_constraint", "b_constraint", "input_dim", "border_mode", "subsample", "dim_ordering", "activity_regularizer"])

        #self.prep_layer also sets slf.__repr__
           
        #self.archicecture_dict gets set in the superclass' constructor, caled with super()
        self._build_architecture(self.architecture_dict)

        
    def train_model(self, X_train, y_train, n_epochs=10):
        self.model.compile(optimizer=self.optimizer, loss={self.output :self.loss})
        return self.model.fit({'input':X_train, self.output:y_train}, nb_epoch=n_epochs)
 
    def classify_batch(self):
        pass

    #===============================================================================
    # private functions

    def _add_input(self, name, input_shape):
        self.model.add_input(name=name, input_shape=input_shape)

    def _add_output(self, name, input_layer):
        """
        input_layer is a string, e.g. "dense1"
        """
        self.model.add_output(name=name, input=input_layer)
        self.output = name

    def _add_layer(self, layer_dict, layer_name, input_layers, merge_mode=None, share_params_with=None):
        """
        input_layers may be either a string or a list.  If it's a list (meaning that there's 
            some loop input), all incoming acivations are merged via merge_mode.
        """
        util.colorprint(layer_name, 'teal')
                        
        layer_dict = dict(layer_dict)
        util.colorprint(layer_dict, 'red')
                        
        if share_params_with is not None:
            print "Warning: ignoring share_params_with"
        
        layer_options = layer_dict["options"]
        layer=None
        if layer_dict["type"]=="conv2d":
            #TODO: remove below
            nb_filter, nb_row, nb_col = 3,3,3
            layer = keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, **layer_options)
        elif layer_dict["type"]=="dense":
            dim  = layer_dict["output_dim"]
            # del layer_options["output_dim"]
            layer = keras.layers.core.Dense(dim, **layer_options)       
        else:
            print "Ursol Major"
            RaiseError()
        # TODO: one of the layers is a string
        if isinstance(input_layers, list):
            #this means that there is input from a loop to this layer
            self.model.add_node(layer, name=layer_name, inputs=input_layers, merge_mode=merge_mode)
        else:
            self.model.add_node(layer, name=layer_name, input=input_layers)

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

    def plot_model(self):
        # grapher = Grapher()
        # grapher.plot(self.model, '%s.png'%self.short_name)
        plotter.plot_model(self.model, '%s_%s.png'%(self.short_name, util.time_string()))



if __name__=="__main__":
    model = LoopyNetwork(architecture_fpath="../architectures/toy_mlp_config.py", n_unrolls=3)

    print repr(model)
    model.plot_model()
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



    model.train_model(X_train, y_train, n_epochs=1000)

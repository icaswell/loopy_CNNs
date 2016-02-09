# loopy_network_lasagne.py
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
import time
from pprint import pprint
sys.path.append("../architectures")

import theano
import theano.tensor as T
import lasagne

from abstract_loopy_network import AbstractLoopyNetwork


class LoopyNetwork(AbstractLoopyNetwork):
    def __init__(self, architecture_fpath, 
                    batch_size,
                    n_unrolls=2, 
                    optimizer = "rmsprop",
                    loss="mse"):
        #===============================================================================
        # Call the superclass init function.  The commented out line is for python 3.
        # super(LoopyNetwork, self).__init__(architecture_fpath, n_unrolls)
        AbstractLoopyNetwork.__init__(self, architecture_fpath, n_unrolls)
        assert self.architecture_dict["framework"] == "lasagne", "Don't try use this on a keras architecture!"

        # unnecessary TODO: have input variable resized in train_model()?
        self.batch_size = batch_size
        self.outputs = {} # mapping from output layer name to the layer that it refers to.
        # self._names_to_layers: a mapping of string (e.g. "layer_1_unroll=2") to a lasagne layer object
        self._names_to_layers = {}

        self._build_architecture(self.architecture_dict)

        
    def train_model(self, X_train, y_train, n_epochs=10):
        """
        """
        #--------------------------------------------------------------------------------------------------
        N = X_train.shape[0]
        if N%self.batch_size:
            print "Warning: batchsize (%s) does not modulo evenly into number of training examples(%s).  %s training examples are being ignored:"%(self.batch_size, N, N - self.batch_size*(N/self.batch_size))

        network = self.network
        loss = self.loss
        # network = self.outputs.values()[1]
        params = lasagne.layers.get_all_params(network, trainable=True)
        #NOTE: assumes only one output
        #TODO: specify learning_rate and momentum elsewhere
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)

        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, self.target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var),
                  dtype=theano.config.floatX)

        input_var = self._names_to_layers["input"].input_var
        train_fn = theano.function([input_var, self.target_var], loss, updates=updates)
        val_fn = theano.function([input_var, self.target_var], [test_loss, test_acc])


        for epoch in range(n_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            #TODO: nee to modify al layer adding to take into account batches????
            for batch in self._iterate_minibatches(X_train, y_train, self.batch_size, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # TODO: uncomment below!
            # And a full pass over the validation data:
            # val_err = 0
            # val_acc = 0
            # val_batches = 0
            # for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            #     inputs, targets = batch
            #     err, acc = val_fn(inputs, targets)
            #     val_err += err
            #     val_acc += acc
            #     val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            # print("  validation accuracy:\t\t{:.2f} %".format(
                # val_acc / val_batches * 100))
 
    def classify_batch(self):
        pass

    #===============================================================================
    # private functions

    def _iterate_minibatches(self, X, y, batchsize, shuffle=False):
        """
        X.shape = (N, input_dim)
        y.shape = (N, output_dim)
        """
        N = X.shape[0]
        order = range(N)
        if shuffle:
            np.random.shuffle(order)

        for i in range(N/batchsize):
            batch_idx = order[i:i + batchsize]
            yield X[batch_idx], y[batch_idx]

    def _add_input(self, name, input_shape):
        input_shape = self.batch_size, input_shape
        input_layer = lasagne.layers.InputLayer(shape=input_shape,
                                    input_var=None)
        # print input_layer.input_var.type() --> <TensorType(float64, matrix)>
        self._names_to_layers[name] = input_layer

    def _add_output(self, name, input_layer):
        """
        input_layer is a string, e.g. "dense1"
        Note that this assumes that there is only one output.  Other outputs will be overridden.
        """
        self.network = self._names_to_layers[input_layer]
        prediction = lasagne.layers.get_output(self.network)
        self.target_var = T.ivector('targets')
        #TODO: get this from self._architecture_dict["layers"]
        loss = lasagne.objectives.categorical_crossentropy(prediction, self.target_var)
        self.loss = loss.mean()
        # self.outputs[name] = loss

    def _convert_layer_options(self, layer_options, layer_type):
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
                            "softmax": lasagne.nonlinearities.softmax,
                    }.get(layer_options["nonlinearity"], None)
            # assert nonlinearity is not None
            if nonlinearity is not None:
                layer_options["nonlinearity"] = nonlinearity

        # self._initialize_params(layer_options, layer_type)
        

    # def _initialize_params(self, layer_options, layer_type):
    #     params_options = {}
        if layer_type=="dense":
            spec = {#TODO: expand to include all types
                # Constant([val]) Initialize weights with constant value.
                # Normal([std, mean]) Sample initial weights from the Gaussian distribution.
                # Uniform([range, std, mean]) Sample initial weights from the uniform distribution.
                # Glorot(initializer[, gain, c01b])   Glorot weight initialization.
                # GlorotNormal([gain, c01b])  Glorot with weights sampled from the Normal distribution.
                # GlorotUniform([gain, c01b]) Glorot with weights sampled from the Uniform distribution.
                # He(initializer[, gain, c01b])   He weight initialization.
                # HeNormal([gain, c01b])  He initializer with weights sampled from the Normal distribution.
                # HeUniform([gain, c01b]) He initializer with weights sampled from the Uniform distribution.
                # Orthogonal([gain])  Intialize weights as Orthogonal matrix.
                # Sparse([sparsity, std]) Initialize weights as sparse matrix.            
                "glorot_uniform": lasagne.init.GlorotUniform('relu')
            }.get(layer_options["W"], None)
            # assert spec is not None
            if nonlinearity is not None:
                # W = lasagne.utils.create_param(spec, shape, name=None)
                layer_options['W'] = spec
            #TODO: change biases as well
        else:
            print "ajystvbkjdfhbvksuydbvwlrtv"*70
            dfkdnfjvsndk

        return layer_options

    def _add_merge_layer(self, layer_name, input_layers):
        """
        TODO: theoretically it would be nice to be able to specify that the mode
        is not just multiplication, but alas time is finite
        """
        merged_name = "\prod {%s}"%input_layers
        layer = lasagne.layers.ElemwiseMergeLayer(incomings=[self._names_to_layers[name] for name in input_layers],
                                                    merge_function=T.mul,
                                                    name=merged_name)
        self._names_to_layers[merged_name] = layer
        return merged_name

    def _add_layer(self, layer_dict, layer_name, input_layers, merge_mode=None, share_params_with=None):
        """
        input_layers may be either a string or a list.  If it's a list (meaning that there's 
            some loop input), all incoming acivations are merged via merge_mode.
        """
        layer_dict = dict(layer_dict)
        # assert isinstance(input_layers, str) #TODO: remove after figuring out layers in lasagne
        if isinstance(input_layers, list):
            input_layers = self._add_merge_layer(layer_name, input_layers)
        
        layer_options = layer_dict["options"]
        pprint(locals())
        #TODO: this shouold be done higher up.  this function is called many times successively.
        layer_options = self._convert_layer_options(layer_options, layer_dict["type"])
        layer=None

        #===============================================================================
        # deal with parametere sharing
        if share_params_with is not None:
            # print dir(self._names_to_layers[share_params_with])
            print self._names_to_layers[share_params_with].get_params()
            layer_options["W"] = self._names_to_layers[share_params_with].W
            layer_options["b"] = self._names_to_layers[share_params_with].b            


        #===============================================================================
        # layer-specific initializations
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
                        name = layer_name,
                        num_units=dim,
                        **layer_options 
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

        self._names_to_layers[layer_name] = layer
        return layer_name      


if __name__=="__main__":
    # model = LoopyNetwork(architecture_fpath="../architectures/toy_loopy_mlp_lasagne_config.py", n_unrolls=1, batch_size=1)
    model = LoopyNetwork(architecture_fpath="../architectures/toy_mlp_config.py", n_unrolls=1, batch_size=24)    

    print repr(model)
    # model.plot_model()
    X_train = np.zeros((0,5))
    y_train_stacked = np.zeros((0,2))
    y_train = np.zeros((0,1))
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
            y_train = np.vstack([y_train, yi])
            y_train_stacked = np.vstack([y_train_stacked, yi_expanded])

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.int32)    
    y_train = y_train[:, 0]



    model.train_model(X_train, y_train, n_epochs=10)

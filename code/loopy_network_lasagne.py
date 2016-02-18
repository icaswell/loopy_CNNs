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
import util


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

        
    def performance_on_whole_set(self, X, y):
        """
        because the batch size is part of the architecture, we can't get the error all in one go.
        Therefore, we have to aggregate it over all the minibatches in something.
        Alas, but a small alas.

        :return: tuple of loss, accuracy
        """
        loss = 0
        acc = 0
        n_batches = 0
        for batch_i, batch in enumerate(self._iterate_minibatches(X, y, self.batch_size, shuffle=False)):
            inputs, targets = batch
            batch_loss, batch_acc = self.val_fn(inputs, targets)
            loss += batch_loss
            acc += batch_acc
            n_batches += 1

        return loss/n_batches, acc/n_batches

      
    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=10, 
                            check_error_n_batches=20,
                            use_expensive_stats=True,
                            check_valid_acc_every=1,
                            ):
        """
        :param bool use_expensive_stats: if this is true, the full training and validation accuracy and 
                loss are calculated every check_error_n_batches batches.
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

        # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001, momentum=0.9)
        updates = lasagne.updates.adam(loss, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, self.target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var),
                  dtype=theano.config.floatX)

        input_var = self._names_to_layers["input"].input_var
        train_fn = theano.function([input_var, self.target_var], loss, updates=updates)
        self.val_fn = theano.function([input_var, self.target_var], [test_loss, test_acc])

        #===============================================================================
        # history: 
        performance_history = {
            "batchly_train_loss": [0.00001],
            "cumulative_train_loss": [],
            "full_train_loss": [],
            "full_train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
        }

        batches_since_last_check = 0


        for epoch in range(n_epochs):
            # In each epoch, we do a full pass over the training data:
            train_loss = 0
            # train_batches = 0
            start_time = time.time()
            #TODO: nee to modify al layer adding to take into account batches????
            for train_batch_i, batch in enumerate(self._iterate_minibatches(X_train, y_train, self.batch_size, shuffle=True)):

                inputs, targets = batch
                # self._print_activations(input_var, inputs)
                # print train_updates_W1(inputs, targets)
                # print train_updates_top_W(inputs, targets)
                batch_loss = train_fn(inputs, targets)
                train_loss += batch_loss
                # train_batches += 1

                #===============================================================================
                # populate performance_history:
                performance_history["batchly_train_loss"][-1] += batch_loss
                if (train_batch_i +1)%check_error_n_batches == 0:
                    print "*"+ "-"*78 + "*"
                    print "Epoch %s, batch %s:"%(epoch, train_batch_i)
                    performance_history["batchly_train_loss"][-1] /= check_error_n_batches
                    performance_history["batchly_train_loss"].append(0.0)
                    performance_history["cumulative_train_loss"].append(train_loss/train_batch_i)

                    print "batchly_train_loss: ", performance_history["batchly_train_loss"][-2]
                    print "cumulative_train_loss: ", performance_history["cumulative_train_loss"][-1]

                    if use_expensive_stats:
                        valid_loss, valid_acc = self.performance_on_whole_set(X_val, y_val)
                        full_train_loss, full_train_acc = self.performance_on_whole_set(X_train, y_train)                        
                        performance_history["valid_loss"].append(valid_loss)
                        performance_history["valid_acc"].append(valid_acc)
                        performance_history["full_train_loss"].append(full_train_loss)
                        performance_history["full_train_acc"].append(full_train_acc)

                        
                        print "valid_loss: ", performance_history["valid_loss"][-1]
                        print "valid_acc: ", performance_history["valid_acc"][-1]
                        print "full_train_loss: ", performance_history["full_train_loss"][-1]
                        print "full_train_acc: ", performance_history["full_train_acc"][-1]



            print "="*80
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_loss / (train_batch_i+1))) #implicitly relies on python scoping, maybe not good style
 
            # And a full pass over the validation data:
            if not epoch%check_valid_acc_every:


                valid_loss, valid_acc = self.performance_on_whole_set(X_val, y_val)
                full_train_loss, full_train_acc = self.performance_on_whole_set(X_train, y_train)     

                # Then we print the results for this epoch:
                print "VALID_LOSS: ", valid_loss
                print "VALID_ACC: ", valid_acc
                print "FULL_TRAIN_LOSS: ", full_train_loss
                print "FULL_TRAIN_ACC: ", full_train_acc
                if not use_expensive_stats:
                    performance_history["valid_loss"].append(valid_loss)
                    performance_history["valid_acc"].append(valid_acc)
                    performance_history["full_train_loss"].append(full_train_loss)
                    performance_history["full_train_acc"].append(full_train_acc)                

        return performance_history

    
    def _print_activations(self, input_var, x_minibatch):
        # print lasagne.layers.get_all_layers(self.network)
        print '-'*80
        print "all layer activations:"
        for layer_name, layer in self._names_to_layers.items():
            self._debug_print("activation for %s:"%layer_name, 0)
            activation = lasagne.layers.get_output(layer).eval({input_var:x_minibatch})
            util.print_matrix(activation, newline=False, color = self.debug_colors[1])


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
        if not hasattr(input_shape, "__iter__"):
            input_shape = [input_shape]
        input_shape = tuple([self.batch_size] + list(input_shape))
        input_layer = lasagne.layers.InputLayer(shape=input_shape,
                                    input_var=None)
        # print input_layer.input_var.type() --> <TensorType(float64, matrix)>
        self._names_to_layers[name] = input_layer

    def _add_output(self, name, input_layer):
        """
        input_layer is a string, e.g. "top_layer" or "dense5"
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
        if layer_type=="dense" or layer_type=="conv2d":
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
            if spec is not None:
                # W = lasagne.utils.create_param(spec, shape, name=None)
                layer_options['W'] = spec
            #TODO: change biases as well
        elif layer_type=="pool2d":
            pass #there are no parameters to initialize
        else:
            print layer_type, " ajystvbkjdfhbvksuydbvwlrtv"*70
            dfkdnfjvsndk

        return layer_options

    def _add_merge_layer(self, layer_name, input_layers, merge_mode):
        """
        TODO: theoretically it would be nice to be able to specify that the mode
        is not just multiplication, but alas time is finite
        """
        merged_name = "\prod {%s}"%input_layers
        merge_fn = {
                "mul":T.mul,
                "sum":T.add,
        }[merge_mode]
        layer = lasagne.layers.ElemwiseMergeLayer(incomings=[self._names_to_layers[name] for name in input_layers],
                                                    merge_function=merge_fn,
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
            input_layers = self._add_merge_layer(layer_name, input_layers, merge_mode=merge_mode)

        
        layer_options = layer_dict["options"]
        #TODO: this shouold be done higher up.  this function is called many times successively.
        layer_options = self._convert_layer_options(layer_options, layer_dict["type"])
        layer=None

        #===============================================================================
        # deal with parameter sharing among equivalent layers in an merge_unroll
        if share_params_with is not None:
            # print dir(self._names_to_layers[share_params_with])
            # print self._names_to_layers[share_params_with].get_params()
            layer_options["W"] = self._names_to_layers[share_params_with].W
            layer_options["b"] = self._names_to_layers[share_params_with].b            


        #===============================================================================
        # layer-specific initializations
        if layer_dict["type"]=="conv2d":
            print layer_options
            layer = lasagne.layers.Conv2DLayer(
                        self._names_to_layers[input_layers], 
                        name = layer_name,
                        **layer_options )
        elif layer_dict["type"]=="dense":
            dim  = layer_dict["output_dim"]

            layer = lasagne.layers.DenseLayer(
                        self._names_to_layers[input_layers], 
                        name = layer_name,
                        num_units=dim,
                        **layer_options 
                    )
        elif layer_dict["type"]=="pool2d":
            layer = lasagne.layers.MaxPool2DLayer(self._names_to_layers[input_layers], 
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

def _make_1d_data_into_fake_image_volume(X):
    N, D = X.shape
    C = 3
    fake_volume = np.zeros((N, C, D, D))
    for i in range(N):
        for j in range(C):
            fake_volume[i, j, :, :] = np.outer(X[i,:], X[i,:])
    return fake_volume


if __name__=="__main__":
    model = LoopyNetwork(architecture_fpath="../architectures/toy_loopy_cnn_lasagne_config.py", n_unrolls=2, batch_size=2)
    # model = LoopyNetwork(architecture_fpath="../architectures/toy_loopy_mlp_lasagne_config.py", n_unrolls=3, batch_size=1)    
    # model = LoopyNetwork(architecture_fpath="../architectures/toy_mlp_config.py", n_unrolls=1, batch_size=24)    

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
    X_train = _make_1d_data_into_fake_image_volume(X_train)
    y_train = y_train.astype(np.int32)    
    y_train = y_train[:, 0]


    X_val = X_train
    y_val = y_train.copy()
    y_val[0:6] = 0



    check_error_n_batches = 2
    history = model.train_model(X_train, y_train, X_val, y_val, n_epochs=50, check_error_n_batches=check_error_n_batches, check_valid_acc_every=4)

    util.plot_loss_acc(history["full_train_loss"], history["full_train_acc"], history["valid_acc"], "batches*%s"%check_error_n_batches, attributes={"lol": 3})

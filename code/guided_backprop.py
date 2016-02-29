# guided_backprop.py


import numpy as np
from collections import defaultdict
import sys
import time
from pprint import pprint
sys.path.append("../architectures")
import cPickle as pickle

import theano
import theano.tensor as T
import lasagne

from loopy_network_lasagne import LoopyNetwork
import util
from guided_backprop_util import *
from data_utils import *

saved_model = "../saved_models/mnist_c3_c3_c1_fc+addition-loop_Feb-27-2016_epoch=19"
# saved_model = "../saved_models/layers=5_loops=1_architecture-ID=10a222a5f3757ea7f2fa6cfafd3a514cdd22d8ca_Feb-20-2016_epoch=25"
model = LoopyNetwork(architecture_fpath="../architectures/mnist_c3_c3_c1_fc+loop.py", n_unrolls=2, batch_size=2)

model.load_model(saved_model)



#===============================================================================
# TODO: export below to function
relu = lasagne.nonlinearities.rectify
relu_layers = [layer for layer in lasagne.layers.get_all_layers(model.network)
               if getattr(layer, 'nonlinearity', None) is relu]
modded_relu = GuidedBackprop(relu)  # important: only instantiate this once!
for layer in relu_layers:
    layer.nonlinearity = modded_relu




#===============================================================================
# 
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()


pprint([layer.name for layer in lasagne.layers.get_all_layers(model.network)])


layer_id = 5
internal_top_layer = lasagne.layers.get_all_layers(model.network)[layer_id]

inp = model.input_var
print inp.shape
outp = lasagne.layers.get_output(internal_top_layer, deterministic=True)
max_outp = T.max(outp)
saliency = theano.grad(max_outp.sum(), wrt=inp)
saliency_function = theano.function([inp], saliency)
# saliency_function = theano.function([inp], [outp])

batch_size = 36

sal =  saliency_function(X_train[0:batch_size])
pprint(sal.nonzero())
print sal.shape






# print dir(internal_top_layer)



# with open(fname, "r") as f:
#     data = pickle.load(f)
# self.network = data["network"]
# self.input_var  = data["input_var"]
# self.target_var  = data["target_var"]
# self.n_pretrained_epochs = data["trained_epochs"]
# self.loss = data["loss"]
# self.compile_model()





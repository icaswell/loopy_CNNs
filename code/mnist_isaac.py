# mnist_isaac.py

import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from data_utils import load_mnist
from loopy_network_lasagne import LoopyNetwork

model = LoopyNetwork(architecture_fpath="../architectures/mnist_isaac_deadweek.py", n_unrolls=1)
print repr(model)

X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

model.train_model(X_train, y_train, X_test, y_test, batchsize=5, n_epochs=10, use_expensive_stats=False)

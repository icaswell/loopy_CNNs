import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from data_utils import load_mnist
from loopy_network_lasagne import LoopyNetwork

model = LoopyNetwork(architecture_fpath="../architectures/mnist_loopy_config.py", n_unrolls=1, batch_size=36)
print repr(model)

X_train, y_train = load_mnist_fail("training", path="../data/mnist")
X_test, y_test = load_mnist_fail("testing", path="../data/mnist")

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

model.train_model(X_train, y_train, X_test, y_test, n_epochs=5, use_expensive_stats=False, check_valid_acc_every=100)

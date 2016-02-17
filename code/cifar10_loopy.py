import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from data_utils import load_cifar10
from loopy_network_lasagne import LoopyNetwork

model = LoopyNetwork(architecture_fpath="../architectures/mnist_loopy_config.py", n_unrolls=1, batch_size=36)
print repr(model)

X_train, y_train, X_val, y_val, X_test, y_test = \
	load_cifar10(num_training=49000, num_validation=1, num_test=1000)

print X_train.shape, y_train.shape
print X_val.shape, y_val.shape

model.train_model(X_train, y_train, X_val, y_val, n_epochs=5, use_expensive_stats=False, check_valid_acc_every=100)

import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from data_utils import load_cifar10
import util
from loopy_network_lasagne import LoopyNetwork

model = LoopyNetwork(architecture_fpath="../architectures/cifar_c3_c5_sm.py", n_unrolls=1)
print repr(model)

X_train, y_train, X_val, y_val, X_test, y_test = \
	load_cifar10(num_training=10000, num_validation=1000, num_test=1000)

print X_train.shape, y_train.shape
print X_val.shape, y_val.shape

history = model.train_model(X_train, y_train, X_val, y_val,batchsize=100,  n_epochs=5, use_expensive_stats=True)

util.plot_loss_acc(history["full_train_loss"], history["full_train_acc"], history["valid_acc"], "batches*%s"%check_error_n_batches, attributes={"lol": 3})

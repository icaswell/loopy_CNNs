import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from data_utils import load_mnist
import util
from loopy_network_lasagne import LoopyNetwork

model = LoopyNetwork(architecture_fpath="../architectures/mnist_nonloopy_config.py", n_unrolls=1, batch_size=50)
print repr(model)

X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

model.train_model(X_train, y_train, X_test, y_test, use_expensive_stats=False, n_epochs=50)

util.plot_loss_acc(history["full_train_loss"], history["full_train_acc"], history["valid_acc"], "batches*%s"%check_error_n_batches, attributes={"lol": 3})

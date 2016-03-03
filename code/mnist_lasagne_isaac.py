# Isaac's playground

import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from data_utils import load_mnist, load_mnist_fail
from loopy_network_lasagne import LoopyNetwork
import util 

# model = LoopyNetwork(architecture_fpath="../architectures/mnist_c3_c5_sm.py", n_unrolls=1, batch_size=36)
model = LoopyNetwork(architecture_fpath="../architectures/mnist_c3_c3_c1_fc+loop.py", n_unrolls=3)
print repr(model)

X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

check_error_n_batches = 300
history = model.train_model(X_train, y_train, X_test, y_test, batchsize=32, n_epochs=20, use_expensive_stats=True, check_error_n_batches=check_error_n_batches)
print history



util.plot_loss_acc(history["full_train_loss"], history["full_train_acc"], history["valid_acc"], "batches*%s"%check_error_n_batches, attributes={"isaac": np.random.rand()})
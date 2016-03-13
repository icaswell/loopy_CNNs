# Isaac's playground

import os, struct, sys
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from data_utils import load_cifar10
from loopy_network_lasagne import LoopyNetwork
import util

# model = LoopyNetwork(architecture_fpath="../architectures/mnist_c3_c5_sm.py", n_unrolls=1, batch_size=36)
#model = LoopyNetwork(architecture_fpath="../architectures/cifar_isaac.py", n_unrolls=3)
sys.setrecursionlimit(100000000)
model = LoopyNetwork(architecture_fpath="../architectures/scq_untied.py",n_unrolls=3, tie_weights=False)
print repr(model)

X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(num_training=20000)


print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

check_error_n_batches = 500
saved_checkpoint = "../saved_models/cifar_c3-32_c3-64_c3-64_c3-1_fc_Mar--4-17:14:12-2016_epoch=8"

# model.load_model(saved_checkpoint)

history = model.train_model(X_train, y_train, X_test, y_test,
                                                batchsize=5,
                                                n_epochs=24,
                                                use_expensive_stats=False,
                                            check_error_n_batches=check_error_n_batches)
print history


util.plot_loss_acc(history["full_train_loss"], history["full_train_acc"], history["valid_acc"], "batches*%s"%check_error_n_batches, attributes={"scq": np.random.rand()})


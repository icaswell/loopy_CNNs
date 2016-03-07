import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from data_utils import load_cifar10
from loopy_network_lasagne import LoopyNetwork
import util 

ARCH_NAME = "cifar_c5-64_pool2_c3-64_c3-64_c3-64_c3-64_c3-64_c3-64_fc"

model = LoopyNetwork(architecture_fpath="../architectures/cifar_resnet_inspired.py", n_unrolls=2)
print repr(model)

X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

check_error_n_batches = 300
        
history = model.train_model(X_train, y_train, X_test, y_test, 
						batchsize=32,
						n_epochs=8,
						use_expensive_stats=False,
					    check_error_n_batches=check_error_n_batches)
print history


util.plot_loss_acc(history["full_train_loss"], history["full_train_acc"], history["valid_acc"], "batches*%s"%check_error_n_batches, attributes={"isaac": np.random.rand()})
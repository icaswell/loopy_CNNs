import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from data_utils import load_mnist
import util
from loopy_network_lasagne import LoopyNetwork

model = LoopyNetwork(architecture_fpath="../architectures/mnist_nonloopy_config.py", n_unrolls=1, batch_size=36)
print repr(model)

X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

print 'X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape)
print 'X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape)
print 'X_test shape: {}, y_test shape: {}'.format(X_test.shape, y_test.shape)
num_test_samples = X_test.shape[0]

model.train_model(X_train, y_train, X_val, y_val, use_expensive_stats=True, n_epochs=10)

util.plot_loss_acc(history["full_train_loss"], history["full_train_acc"], history["valid_acc"], "batches*%s"%check_error_n_batches, attributes={"lol": 3})

# evaluate on test set
print 'Testing on a set of {} samples'.format(num_test_samples)
test_loss, test_acc = model.performance_on_whole_set(X_test, y_test)

print 'Test loss: {}, test accuracy: {}'.format(test_loss, test_acc)


# Isaac's playground

import os, struct, sys
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from data_utils import load_cifar10
from loopy_network_lasagne import LoopyNetwork
import util

sys.setrecursionlimit(100000000)
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(num_training=20000)

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

models = {}
models['vanilla-5'] = ['../saved_models/scq_unrolled_unrolls=1_Mar-13-07:38:46-2016_epoch=23', 1, True]
models['loopy-5'] = ['../saved_models/scq_unrolled_unrolls=3_Mar-12-19:56:13-2016_epoch=23', 3, True]
models['deep-5'] = ['../saved_models/scq_untied_unrolls=3_Mar-12-13:08:13-2016_epoch=22', 3, False]
models['vanilla-32'] = ['../saved_models/cifar_c3-128_c3-128_c3-3_fc_sumloop_nounroll_unrolls=1_Mar-11-02:33:47-2016_epoch=23',1 , True]
models['loopy-32'] = ['../saved_models/cifar_c3-128_c3-128_c3-3_fc_sumloop_nounroll_unrolls=3_Mar-12-06:37:24-2016_epoch=14', 3, True]
models['deep-32'] = ['../saved_models/cifar_c3-128_c3-128_c3-3_fc_sumloop_nounroll_unrolls=3_Mar-12-06:38:07-2016_epoch=14', 3, False]
models['xloop-5'] = ['../saved_models/deadweek_multiplication_loop_unrolls=3_Mar-13-15:03:52-2016_epoch=23', 3, True]
models['+loop-5'] = ['../saved_models/cifar_c3-128_c3-128_c3-3_fc_sumloop_nounroll_unrolls=3_Mar-13-07:22:43-2016_epoch=23', 3, True]

results = {}

for name, vals in models.iteritems():
	print "Testing " + name
	model = LoopyNetwork(architecture_fpath="../architectures/scq_unrolled.py",n_unrolls=vals[1], tie_weights=vals[2])
	model.load_model(vals[0])
	loss, acc = model.performance_on_whole_set(X_val, y_val)
	results[name] = acc

for name, acc in results.iteritems():
	print name + ": ", acc

from guided_backprop import save_filter_visualizations_to_folder
from data_utils import load_cifar10
from loopy_network_lasagne import LoopyNetwork

#===============================================================================
# Sample script
# X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

# saved_model = "../saved_models/mnist_c3_c3_c1_fc+addition-loop_Feb-27-2016_epoch=19"
# saved_model = "../saved_models/layers=5_loops=1_architecture-ID=10a222a5f3757ea7f2fa6cfafd3a514cdd22d8ca_Feb-20-2016_epoch=25"
# arch_fpath = "../architectures/mnist_c3_c3_c1_fc+loop.py"
N_IMAGES = 20
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
arch_fpath = "../architectures/cifar_resnet_inspired.py"
saved_model = "../saved_models/cifar_c5-64_pool2_c3-64_c3-64_c3-64_c3-64_c3-64_c3-64_fc_Mar--5-2016_epoch=7"
model = LoopyNetwork(architecture_fpath=arch_fpath, n_unrolls=2)
model.load_model(saved_model)

# # loss_train, acc_train = model.performance_on_whole_set(X_train, y_train)
# loss_val, acc_val = model.performance_on_whole_set(X_val, y_val)
# loss_test, acc_test = model.performance_on_whole_set(X_test, y_test)

# # print "Training loss: {} \tacc: {}".format(loss_train, acc_train)
# print "Validation loss: {} \tacc: {}".format(loss_val, acc_val)
# print "Test loss: {} \tacc: {}".format(loss_test, acc_test)


filters_per = 5 #visualize this many filters
filters_to_vis = {

	# 'conv_1_unroll=0': range(filters_per),
 # 	'conv_1_unroll=1': range(filters_per),
 # 	'conv_2_unroll=0': range(filters_per),
 # 	'conv_2_unroll=1': range(filters_per),
 # 	'conv_3_unroll=0': range(filters_per),
 # 	'conv_3_unroll=1': range(filters_per),
 # 	'conv_4_unroll=0': range(filters_per),
 # 	'conv_4_unroll=1': range(filters_per),
 # 	'conv_5_unroll=0': range(filters_per),
 # 	'conv_5_unroll=1': range(filters_per),
 # 	'conv_6_unroll=0': range(filters_per),
 # 	'conv_6_unroll=1': range(filters_per),
 # 	'conv_7_unroll=0': range(filters_per),
 # 	'conv_7_unroll=1': range(filters_per),
 	'loop_layer_1_unroll=0': range(filters_per),
 	'loop_layer_1_unroll=1': range(filters_per),
 	'loop_layer_2_unroll=0': range(filters_per),
 	'loop_layer_2_unroll=1': range(filters_per),

 }

#===============================================================================
# Now actually visualize
save_filter_visualizations_to_folder(model, dirname="../pictures", 
											X=X_train[0:N_IMAGES], 
											layers_to_vis=filters_to_vis.keys(), 
											filters_to_vis=filters_to_vis, 
											run_stem="gbp_cifar",
											positivize_gradients=True)
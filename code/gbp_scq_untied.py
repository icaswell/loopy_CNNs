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
arch_fpath = "../architectures/scq_unrolled.py"
saved_model = "../saved_models/scq_untied_unrolls=3_Mar-12-13:08:13-2016_epoch=22"
model = LoopyNetwork(architecture_fpath=arch_fpath, n_unrolls=3)
model.load_model(saved_model)

filters_per = 2 #visualize this many filters
filters_to_vis = {
 'conv_1_unroll=0': range(filters_per),
 'conv_1_unroll=1': range(filters_per),
 'conv_1_unroll=2': range(filters_per),
 'conv_1_unroll=3': range(filters_per),
 'conv_2_unroll=0': range(filters_per),
 'conv_2_unroll=1': range(filters_per),
 'conv_2_unroll=2': range(filters_per),
 'conv_2_unroll=3': range(filters_per),
 'conv_3_unroll=0': range(filters_per),
 'conv_3_unroll=1': range(filters_per),
 'conv_3_unroll=2': range(filters_per),
 'conv_3_unroll=3': range(filters_per),
 # "\\prod {['input', 'conv_4_unroll=0']}"
 }

#===============================================================================
# Now actually visualize
save_filter_visualizations_to_folder(model, dirname="../pictures", 
											X=X_train[0:N_IMAGES], 
											layers_to_vis=filters_to_vis.keys(), 
											filters_to_vis=filters_to_vis, 
											run_stem="gbp_scq_untied",
											positivize_gradients=True)

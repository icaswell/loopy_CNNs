# File: toy_loopy_mlp_lasagne_config.py
# @author: Isaac Caswell
# @created: Feb 7 2016
#

{
"run_params":
        {"use_batchnorm":True},
"framework":"lasagne",
"name": "scq_untied",

"templates":
        {
        "input": {"type": "input"},
        "conv3": {"type": "conv2d", "W": "glorot_uniform", "filter_size": 3, "pad": 1},
        "conv5": {"type": "conv2d", "W": "glorot_uniform", "filter_size": 5, "pad": 2},
        "pool": {"type": "pool2d", "pool_size":2},
        "dense": {"type": "dense", "W": "glorot_uniform"},
        },

#===============================================================================
# Here's where you define your layers.  You'll wire them together later.

"layers": #parser asserts that there exists a layer called "input" and a layer called "output"
        {
        "input":{"output_dim":(3,32,32), "template": "input"},
        "conv_1":{"num_filters":64, "template": "conv3"},
        "conv_2":{"num_filters":64, "template": "conv3"},
        "conv_3":{"num_filters":3, "template": "conv3"},
        "fc_1":{"output_dim":10, "template": "dense", "nonlinearity":"softmax"},
        },

"stacks":
        {
        "main_stack": {
                "type": "main",
                "structure": ["input", "conv_1", "conv_2", "conv_3", "fc_1"]
                },
         "loop": {
                "type": "loop",
                "structure": ["conv_3", "conv_1"],
                "composition_mode": "sum"
                },

        },

"layer_defaults":
        {
        "dense":
                {
                "W": "glorot_uniform",
                "nonlinearity": "relu"
                },
        "conv2d":
                {
                "W": "glorot_uniform",
                "num_filters": 3,
                "filter_size": 3,
                "pad": 1,
                "nonlinearity": "relu"
                },

                # class lasagne.layers.MaxPool2DLayer(incoming, pool_size, stride=None, pad=(0, 0), ignore_border=True, **kwargs)[source]
        "pool2d":
                {
                }
        }
}


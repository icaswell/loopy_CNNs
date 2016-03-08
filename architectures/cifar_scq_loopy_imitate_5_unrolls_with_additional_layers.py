# Lisa Wang
# @created:  Mar 7 2016

# This architecture does not have any loops. It simulates the depth of 
# cifar_scq_loopy with n_unrolls=5, by adding in new layers for each unroll
# This model is created to compare performance between similar architectures 
# which differ vastly in the number of parameters. 

{
"run_params":
        {"use_batchnorm":True},
"framework":"lasagne",
"name": "cifar_scq_loopy",

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
        "conv_1a":{"num_filters":48, "template": "conv3"},
        "conv_2a":{"num_filters":48, "template": "conv3"},
        "conv_3a":{"num_filters":48, "template": "conv3"},
        "conv_4a":{"num_filters":48, "template": "conv3"},
        "conv_5a":{"num_filters":3, "template": "conv3"},
        "conv_1b":{"num_filters":48, "template": "conv3"},
        "conv_2b":{"num_filters":48, "template": "conv3"},
        "conv_3b":{"num_filters":48, "template": "conv3"},
        "conv_4b":{"num_filters":48, "template": "conv3"},
        "conv_5b":{"num_filters":3, "template": "conv3"},
        "conv_1c":{"num_filters":48, "template": "conv3"},
        "conv_2c":{"num_filters":48, "template": "conv3"},
        "conv_3c":{"num_filters":48, "template": "conv3"},
        "conv_4c":{"num_filters":48, "template": "conv3"},
        "conv_5c":{"num_filters":3, "template": "conv3"},
        "conv_1d":{"num_filters":48, "template": "conv3"},
        "conv_2d":{"num_filters":48, "template": "conv3"},
        "conv_3d":{"num_filters":48, "template": "conv3"},
        "conv_4d":{"num_filters":48, "template": "conv3"},
        "conv_5d":{"num_filters":3, "template": "conv3"},
        "conv_1e":{"num_filters":48, "template": "conv3"},
        "conv_2e":{"num_filters":48, "template": "conv3"},
        "conv_3e":{"num_filters":48, "template": "conv3"},
        "conv_4e":{"num_filters":48, "template": "conv3"},
        "conv_5e":{"num_filters":3, "template": "conv3"},
        "fc_1":{"output_dim":10, "template": "dense", "nonlinearity":"softmax"},
        },

"stacks":
        {
        "main_stack": {
                "type": "main",
                "structure": [
                    "input", 
                    "conv_1a",
                    "conv_2a",
                    "conv_3a",
                    "conv_4a",
                    "conv_5a",
                    "conv_1b",
                    "conv_2b",
                    "conv_3b",
                    "conv_4b",
                    "conv_5b",
                    "conv_1c",
                    "conv_2c",
                    "conv_3c",
                    "conv_4c",
                    "conv_5c",
                    "conv_1d",
                    "conv_2d",
                    "conv_3d",
                    "conv_4d",
                    "conv_5d",
                    "conv_1e",
                    "conv_2e",
                    "conv_3e",
                    "conv_4e",
                    "conv_5e",
                     "fc_1"
                    ]
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


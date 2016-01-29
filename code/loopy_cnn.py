# loopy_cnn.py
# @author: Isaac Caswell
# @created: Jan 28 2016
#
#===============================================================================
# DESCRIPTION:
#
# Defines the LoopyCNN class.  It is designed to build architecture that's defined
# in a config file.  (see architecture_config_readme.py for an example)
#
#===============================================================================
# CURRENT STATUS: Massively Unimplemented
#===============================================================================
# USAGE:
# from loopy_cnn import LoopyCNN
# model = LoopyCNN(architecture_fpath="../architectures/simple_loop.py", 
#         **kwargs)




import sys
sys.path.append("../architectures")

from architecture_config_asserter import sanity_check
 

class LoopyCNN():
    def __init__(self, architecture_fpath, n_unrolls=2):

        self.description = "uninitialized LoopyCNN instance"        
        self.n_unrolls = n_unrolls

        self._build_architecture(architecture_fpath)
        self.params = {}
 
    def train_model(self):
        pass
 
    def classify_batch(self):
        pass
 
    def _build_architecture(self, architecture_fpath):
        with open(architecture_fpath, "r") as f:
            string_representation = f.read()
        architecture_dict = eval(string_representation)
 
        # sanity check: make sure that the architecture is consistent
        # and wel configured.
        sanity_check(architecture_dict)
 
        self._build_description(architecture_dict)

 
 
         # all_hidden_activations = []
         # if self.debug: util.colorprint("building architecture...", self.debug_model_color)
         # # loop_inputs = a mapping from index to vector (dict), initialized to nothing
         # # loop_outputs = a mapping from index to vector (dict), initialized to ones
 
         # loop_inputs = {}
         # loop_outputs = defaultdict(list)
 
         # #======================================================
         # # note that "output_layer" means output FROM the loop, and 
         # # input_layer means input TO the loop.  Perhaps better names are in order.
         # for input_layer, output_layer in self.loops:
         #     loop_inputs[input_layer] = None
         #     loop_output_shape = (self.all_layer_dims[output_layer], 1)
         #     loop_outputs[output_layer].append(np.ones(loop_output_shape))
         #     if self.debug:
         #         util.colorprint("creating loop output of shape %s from layer %s to layer %s"%(loop_output_shape, input_layer, output_layer), self.debug_model_color)
 
         # for _ in range(self.n_unrolls):
         #     hidden_activation = input
         #     # hidden_activation = self._prepend_intercept(input)            
         #     #====================================================
         #     # put in all the weight matrices and populate the inputs to the loops
         #     for layer_i, w_name in enumerate(w_names):
         #         #optional addendum: terminate when the last loop is reached, unless it's the last unroll
         #         # b_name = self._b_name_from_w_name(w_name)
         #         if self.debug:
         #             util.colorprint("Inserting parameters %s and (layer %s) into the graph..."%(w_name, layer_i), self.debug_model_color)
 
         #         if layer_i in loop_outputs:
         #             for loop_output in loop_outputs[layer_i]:
         #                 if self.debug:
         #                     util.colorprint("\tInserting incoming loop activation...", self.debug_model_color)
         #                 hidden_activation *= loop_output
         #             loop_outputs[layer_i] = []
 
         #         hidden_activation = T.dot(self.tparams[w_name], hidden_activation) 
 
         #         # print (1, hidden_activation.shape[1])
         #         # hidden_activation += T.tile(self.tparams[b_name], (1, hidden_activation.shape[1]))
         #         # hidden_activation += self.tparams[b_name]
         #         # hidden_activation = T.tanh(hidden_activation)
         #         hidden_activation = T.nnet.sigmoid(hidden_activation)
         #         # hidden_activation = self._prepend_intercept(hidden_activation)
 
         #         #---------------------------------------------------
         #         if layer_i in loop_inputs:
         #             if self.debug:
         #                 util.colorprint("\tStoring outgoing loop activation...", self.debug_model_color)
         #             loop_inputs[layer_i] = hidden_activation
 
         #         #---------------------------------------------------                    
         #         all_hidden_activations.append(hidden_activation)
         #     #====================================================
         #     # calculate the outputs 
         #     for u_i, u_name in enumerate(u_names):
         #         input_layer, output_layer = self.loops[u_i]
         #         # b_name = self._b_name_from_w_name(u_name)
         #         if self.debug:
         #             util.colorprint("inserting %s and %s into the graph, ready to feed into layer %s"%(u_name, b_name, output_layer), self.debug_model_color)
         #         loop_output = T.dot(self.tparams[u_name], loop_inputs[input_layer])# +  self.tparams[b_name]
         #         loop_output = T.nnet.sigmoid(loop_output)
         #         loop_outputs[output_layer].append(loop_output)
 
         # # final_activation = all_hidden_activations[-1]
         # self.final_activation = T.nnet.softmax(all_hidden_activations[-1].T)
         # all_hidden_activations.append(self.final_activation)
 
         # self.all_hidden_activations = all_hidden_activations
 
         # # self.final_activation = 1.0/(1.0 + T.nnet.sigmoid(final_activation))
 
         # # off = 1e-8
         # # if final_activation.dtype == 'float16':
         # #     off = 1e-6
 
         # # self.cost = -T.log(final_activation[self.y, 0] + off)        
        
         # cost = self.loss_function(self.final_activation, self.y) + self.L1_reg * self.L1
 
         # return cost       

    def _build_description(self, architecture_dict):
        title_color = 95
        layer_colors = {
                "input": 93,
                "conv": 96,
                "fully_connected": 97,
        }
        self.description = "LoopyCNN instance with the following hyperparameters, layers and loops:"
        self.description += "\033[%sm"%title_color + "\nHYPERPARAMETERS:" + '\033[0m'
        self.description += "\n\tn_unrolls=%s"%self.n_unrolls
        self.description += "\033[%sm"%title_color + "\n\nARCHITECTURE:" + '\033[0m'
        for stack_name, stack in architecture_dict["stacks"].items():
            self.description +="\n%s:"%stack_name
            for layer_name in stack["structure"]:
                layer = architecture_dict["layers"][layer_name]
                if "template" in layer:
                    template = architecture_dict["templates"][layer["template"]]
                    template.update(layer)
                    layer=template
                # TODO: replace this with looping through the actual layers, each of which will have a __str__() method
                layer_desc = layer_name
                layer_desc += " [%s layer; dim=%s]"%(layer["type"], layer["dim"])
                layer_color = layer_colors[layer["type"]]
                self.description += "\n\t\033[%sm"%layer_color + layer_desc + '\033[0m'



    def __repr__(self):
        """
        returns a descriptive string representation of the model.
        """
        return self.description


if __name__=="__main__":
     model = LoopyCNN(architecture_fpath="../architectures/architecture_config_readme.py")

     print repr(model)


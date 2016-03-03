import pydot

def plot_model(model, to_file='model.png'):

    graph = pydot.Dot(graph_type='digraph')

    # don't need to append number for names since all nodes labeled
    for input_node in model.input_config:
        graph.add_node(pydot.Node(input_node['name']))

    # intermediate and output nodes have input defined
    for layer_config in [model.node_config, model.output_config]:
        for node in layer_config:
            graph.add_node(pydot.Node(node['name']))
            # possible to have multiple 'inputs' vs 1 'input'
            if node['inputs']:
                for e in node['inputs']:
                    graph.add_edge(pydot.Edge(e, node['name']))
            else:
                graph.add_edge(pydot.Edge(node['input'], node['name']))

    graph.write_png(to_file)

if __name__=="__main__":
    from loopy_network_lasagne import LoopyNetwork
    import lasagne

    # model = LoopyNetwork(architecture_fpath="../architectures/mnist_c3_c5_sm.py", n_unrolls=1, batch_size=36)
    model = LoopyNetwork(architecture_fpath="../architectures/mnist_c3_c3_c1_fc+loop.py", n_unrolls=2, batch_size=36)
    print repr(model)
    layas =  lasagne.layers.get_all_layers(model.network)
    print [l.name for l in layas]
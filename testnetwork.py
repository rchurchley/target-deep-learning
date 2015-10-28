import sys
import lasagne
from deepsix.experiment import Experiment


def network(input_var=None):
    network = lasagne.layers.InputLayer(
        shape=(None, 3, 64, 64),
        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=8, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=8, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 path/to/data_dir path/to/experiment_dir')
        exit()
    exp = Experiment(data=sys.argv[1], directory=sys.argv[2], network=network)
    exp.load_parameters()
    exp.test()

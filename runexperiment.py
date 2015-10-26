import sys
import theano
import lasagne
from deepsix.experiment import Experiment


def network(input_var=None):
    network = lasagne.layers.InputLayer(
        shape=(None, 3, 64, 64),
        input_var=input_var)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 path/to/data_dir path/to/output_dir n_epochs')
        exit()
    exp = Experiment(data=sys.argv[1], directory=sys.argv[2], network=network)
    n = 10 if len(sys.argv) == 3 else int(sys.argv[3])
    exp.train(epochs=n)
    exp.save()

import theano
import lasagne
import numpy
import uuid
from lasagne.layers import get_output, get_all_params
from deepsix.train import *

input_directory = 'data/example'
num_epochs = 100


def load_datasets(input_directory):
    """Return the training, validation, and testing data and labels."""
    result = {'training': {}, 'validation': {}, 'testing': {}}
    for dataset in result:
        for part in ['data', 'labels']:
            result[dataset][part] = numpy.load(
                '{}/{}_{}.npy'.format(input_directory, dataset, part))
    return result


def build_network(input_var=None):
    network = lasagne.layers.InputLayer(
        shape=(None, 3, 64, 64),
        input_var=input_var)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def build_model(input_var=None, target_var=None, network=None):
    def loss(test=False):
        return lasagne.objectives.categorical_crossentropy(
            get_output(network, deterministic=test), target_var).mean()

    optimizer = lasagne.updates.nesterov_momentum(
        loss(),
        get_all_params(network, trainable=True),
        learning_rate=0.001,
        momentum=0.1)

    return {
        'input_var': input_var,
        'target_var': target_var,
        'network': network,
        'loss': loss,
        'updates': optimizer
    }


def main(input_directory, num_epochs=5):
    print("Loading data...")
    data = load_datasets(input_directory)

    print("Building model...")
    input_var = theano.tensor.tensor4('inputs')
    target_var = theano.tensor.ivector('targets')
    network = build_network(input_var)
    model = build_model(input_var, target_var, network)

    train_network(data, model, num_epochs)

    print("\nLearned parameters:")
    params = numpy.array(lasagne.layers.get_all_param_values(network))
    param_file = 'output/{}.npy'.format(str(uuid.uuid4()))
    print("Parameters written to {}".format(param_file))
    params.dump(param_file)


if __name__ == '__main__':
    main(input_directory, num_epochs)

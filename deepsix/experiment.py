import numpy
import theano
import theano.tensor
import lasagne
from lasagne.layers import get_output, get_all_params
import time
import os
import inspect
import json
import csv


class Experiment:
    """A class for conducting experiments with neural networks.

    Attributes:
        training,
        validation,
        testing: Pairs, each consisting of a numpy array containing pixel data
            for many images and a numpy vector containing the labels of each
            image.
        report: A dictionary of strings and parameters describing the
            experiment, including the network structure and training process.
        history: A list recording the training and validation loss and
            accuracy for each epoch in the training process.
    """

    def __init__(self, data, directory, network, **kwargs):
        """Load datasets and compile the neural network model.

        Args:
            data: The path of the directory containing the dataset .npy files.
            directory: The directory path to save the output of the experiment.
            network: A function taking a Theano input variable and returning
                the output layer of a Lasagne neural network.
            **kwargs: Passed to self.__compile_model.
        """
        os.makedirs(directory, exist_ok=True)  # Ensure `directory` exists
        self.directory = directory
        self.report = {}
        self.history = [['Epoch', 'Trn_Err', 'Trn_Acc', 'Val_Err', 'Val_Acc']]
        self.__compile_model(network, **kwargs)
        self.__load_data(data)

    def train(self, epochs):
        """Train the network, report on progress, and output test results.

        Args:
            epochs: Number of epochs to train for.
        """
        print('Starting training...')
        print('\n{:4} {:>17}  {:>17}'.format('', 'Training', 'Validation'))
        print('{:4} {:>8} {:>8}  {:>8} {:>8} {:>8}'.format(
            '', 'Loss', 'Acc', 'Loss', 'Acc', 'Time'))
        training_time = 0
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            trn_err, trn_acc = self.__progress(self.training, self.__train_fn)
            val_err, val_acc = self.__progress(self.validation, self.__val_fn)
            elapsed_time = time.time() - start_time
            training_time += elapsed_time
            print('{:>4} {:>8.3f} {:>8.1%}  {:>8.3f} {:>8.1%}  {:>7.2f}s'
                  ''.format(epoch, trn_err, trn_acc, val_err, val_acc,
                            elapsed_time))
            self.history.append([epoch, trn_err, trn_acc, val_err, val_acc])
        tst_err, tst_acc = self.__progress(self.testing, self.__val_fn)
        print('Test loss: {}'.format(tst_err))
        print('Test accuracy: {:.3%}'.format(tst_acc))
        self.report['epochs'] = epochs
        self.report['time_per_epoch'] = training_time / epochs
        self.report['test_loss'] = tst_err
        self.report['test_accuracy'] = tst_acc

    def save(self):
        """Save the results of the experiment to self.directory."""
        filename = os.path.join(self.directory, 'experiment.json')
        with open(filename, 'w') as f:
            json.dump(self.report, f, indent=2, sort_keys=True)
        filename = os.path.join(self.directory, 'training_progress.csv')
        with open(filename, 'w') as csvfile:
            csv.writer(csvfile).writerows(self.history)
        filename = os.path.join(self.directory, 'learned_parameters.npy')
        parameters = lasagne.layers.get_all_param_values(self.__network)
        numpy.array(parameters).dump(filename)

    def __compile_model(self,
                        network,
                        loss=lasagne.objectives.categorical_crossentropy,
                        learning_rate=0.001,
                        momentum=0.1):
        """Compile the Theano functions used in the experiment."""
        print('Compiling model...')
        self.report['network'] = inspect.getsource(network)
        self.report['loss_function'] = loss.__name__
        self.report['learning_rate'] = learning_rate
        self.report['learning_momentum'] = momentum
        start_time = time.time()
        self.__input_var = theano.tensor.tensor4('inputs')
        self.__target_var = theano.tensor.ivector('targets')
        self.__network = network(self.__input_var)
        self.__loss = lambda t: loss(get_output(self.__network,
                                                deterministic=t),
                                     self.__target_var).mean()
        self.__optimizer = lasagne.updates.nesterov_momentum(
            self.__loss(False),  # enable dropout during training
            get_all_params(self.__network, trainable=True),
            learning_rate=learning_rate,
            momentum=momentum)
        accuracy = theano.tensor.mean(
            theano.tensor.eq(
                theano.tensor.argmax(
                    get_output(
                        self.__network,
                        deterministic=True),
                    axis=1),
                self.__target_var),
            dtype=theano.config.floatX)
        self.__train_fn = theano.function(
            [self.__input_var, self.__target_var],
            [self.__loss(False), accuracy],
            updates=self.__optimizer)
        self.__val_fn = theano.function(
            [self.__input_var, self.__target_var],
            [self.__loss(True), accuracy])
        elapsed_time = time.time() - start_time
        self.report['time_to_compile'] = elapsed_time

    def __load_data(self, input_directory):
        """Initialize data sets from .npy files in self.directory.

        Each of self.training, self.validation, and self.testing is
        initialized as a pair: the first element a numpy array contains image
        data, and the second element a numpy vector containing the label
        corresponding to each image.
        """
        print("Loading data...")
        self.training, self.validation, self.testing = (
            tuple(
                numpy.load(
                    os.path.join(input_directory, '{}_{}.npy'.format(x, y))
                )
                for y in ('data', 'labels')
            )
            for x in ('training', 'validation', 'testing')
        )
        self.report['data_directory'] = input_directory
        self.report['images_training'] = len(self.training[1])
        self.report['images_validation'] = len(self.validation[1])
        self.report['images_testing'] = len(self.validation[1])

    def __progress(self, dataset, input_function):
        """Train network for one epoch (one pass through all minibatches).

        Return:
            Average objective function value and accuracy.
        """
        total_err = 0
        total_acc = 0
        total_batches = 0
        for batch in self.__iterate_minibatches(dataset, batchsize=100):
            inputs, targets = batch
            err, acc = input_function(inputs, targets)
            total_err += err
            total_acc += acc
            total_batches += 1
        return total_err / total_batches, total_acc / total_batches

    def __iterate_minibatches(self, dataset, batchsize, shuffle=False):
        """Split input data into batches and iterate over the minibatches."""
        n = len(dataset[0])
        assert len(dataset[1]) == n
        batchsize = min(batchsize, n)
        indices = numpy.arange(n)
        numpy.random.shuffle(indices)
        for start_idx in range(0, n - batchsize + 1, batchsize):
            excerpt = indices[start_idx:start_idx + batchsize]
            yield dataset[0][excerpt], dataset[1][excerpt]

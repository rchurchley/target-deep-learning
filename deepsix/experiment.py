import numpy
import theano
import theano.tensor as T
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
        if not os.path.exists(directory):
            os.makedirs(directory)  # Ensure `directory` exists
        self.directory = directory
        self.report = {}
        self.history = [['Epoch',
                         'Trn_Loss', 'Trn_Prc', 'Trn_Rec', 'Trn_Acc',
                         'Val_Loss', 'Val_Prc', 'Val_Rec', 'Val_Acc']]
        self.__compile_model(network, **kwargs)
        self.__load_data(data)

    def load_parameters(self, filename=None):
        """Load the learned parameters from a previous experiment."""
        if not filename:
            filename = os.path.join(self.directory, 'learned_parameters.npy')
        params = numpy.load(filename)
        lasagne.layers.set_all_param_values(self.__network, params)

    def train(self, epochs):
        """Train the network, report on progress, and output test results.

        Args:
            epochs: Number of epochs to train for.
        """
        print('Starting training...')
        print('\n{:13} '
              '{:>17}  '
              '{:^38}'
              ''.format('', '--- Training ---', '--- Validation ---'))
        print('{:4} {:>8} '
              '{:>8} {:>8}  '
              '{:>8} {:>8} {:>8} {:>8}'
              ''.format('', '', 'Loss', 'Acc', 'Loss', 'Prc', 'Rec', 'Acc'))
        training_time = 0
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            trn_stats = self.__progress(self.training, self.__train_fn)
            val_stats = self.__progress(self.validation, self.__val_fn)
            elapsed_time = time.time() - start_time
            training_time += elapsed_time
            print('{:>4} {:>7.2f}s '
                  '{:>8.3f} {:>8.1%}  '
                  '{:>8.3f} {:>8.1%} {:>8.1%} {:>8.1%}'
                  ''.format(epoch, elapsed_time,
                            trn_stats[0], trn_stats[-1],
                            *val_stats))
            self.history.append([epoch] + list(trn_stats) + list(val_stats))
        self.report['epochs'] = epochs
        self.report['time_per_epoch'] = training_time / epochs

    def test(self):
        """Test the learned parameters on the testing dataset."""
        statistics = self.__progress(self.testing, self.__val_fn)
        print('Loss:      {}'.format(statistics[0]))
        print('Precision: {:.3%}'.format(statistics[1]))
        print('Recall:    {:.3%}'.format(statistics[2]))
        print('Accuracy:  {:.3%}'.format(statistics[3]))
        self.report['test_loss'] = statistics[0]
        self.report['test_precision'] = statistics[1]
        self.report['test_recall'] = statistics[2]
        self.report['test_accuracy'] = statistics[3]

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
        parameters = parameters
        numpy.save(filename, parameters)

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
        self.__input_var = T.tensor4('inputs')
        self.__target_var = T.ivector('targets')
        self.__network = network(self.__input_var)
        self.__loss = lambda t: loss(get_output(self.__network,
                                                deterministic=t),
                                     self.__target_var).mean()
        self.__optimizer = lasagne.updates.nesterov_momentum(
            self.__loss(False),  # enable dropout during training
            get_all_params(self.__network, trainable=True),
            learning_rate=learning_rate,
            momentum=momentum)
        predictions = T.argmax(
            get_output(self.__network, deterministic=True),
            axis=1)
        # number of correct predictions
        n_correct = T.sum(T.eq(predictions, self.__target_var))
        # number of relevant images in the sample
        n_relevant = T.sum(self.__target_var)
        # number of images predicted to be relevant
        n_selected = T.sum(predictions)
        # number of correct predictions of relevance
        n_correct_relevant = T.sum(predictions & self.__target_var)
        statistics = [n_correct, n_selected, n_relevant, n_correct_relevant]
        self.__train_fn = theano.function(
            [self.__input_var, self.__target_var],
            [self.__loss(False)] + statistics,
            updates=self.__optimizer)
        self.__val_fn = theano.function(
            [self.__input_var, self.__target_var],
            [self.__loss(True)] + statistics)
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
        batchsize = 100
        total_loss = 0
        total_correct = 0
        total_selected = 0
        total_relevant = 0
        total_correct_relevant = 0
        total_batches = 0
        for batch in self.__iterate_minibatches(dataset, batchsize):
            inputs, targets = batch
            l, c, s, r, h = input_function(inputs, targets)
            total_loss += l
            total_correct += c
            total_selected += s
            total_relevant += r
            total_correct_relevant += h
            total_batches += 1
        avg_loss = total_loss / total_batches
        precision = total_correct_relevant / total_selected
        recall = total_correct_relevant / total_relevant
        accuracy = total_correct / (batchsize * total_batches)
        return avg_loss, precision, recall, accuracy

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

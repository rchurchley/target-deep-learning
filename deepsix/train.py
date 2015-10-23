import numpy
import theano
import theano.tensor
from lasagne.layers import get_output
import time


def compile_model(model):
    """Compile Theano functions for learning process.

    Args:
        model (dict): Dictionary with the following contents.
            'input_var': Theano input variable.
            'target_var': Theano target variable.
            'network': output layer of Lasagne network.
            'loss': Lasagne objective function.
            'updates': optimizer from Lasagne.updates.

    Return:
        The network and references to the Theano-compiled training and
        validation methods.
    """
    network = model['network']

    test_acc = theano.tensor.mean(
        theano.tensor.eq(
            theano.tensor.argmax(
                get_output(
                    network,
                    deterministic=True),
                axis=1),
            model['target_var']),
        dtype=theano.config.floatX)

    # Compile functions
    train_fn = theano.function(
        [model['input_var'], model['target_var']],
        model['loss'](),
        updates=model['updates'])

    val_fn = theano.function(
        [model['input_var'], model['target_var']],
        [model['loss'](test=True), test_acc])

    return network, train_fn, val_fn


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """Split input data into batches and iterate over the minibatches."""
    assert len(inputs) == len(targets)
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def full_pass(input_dict, val_fn):
    """Train network for one epoch (one pass through all minibatches).

    Return:
        Average objective function value and accuracy.
    """
    total_err = 0
    total_acc = 0
    total_batches = 0
    for batch in iterate_minibatches(input_dict['data'],
                                     input_dict['labels'],
                                     1,
                                     shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        total_err += err
        total_acc += acc
        total_batches += 1
    return total_err / total_batches, total_acc / total_batches


def train_network(inputs, model, num_epochs):
    """Train a Lasagne network, report on progress, and output test accuracy.

    Args:
        inputs (dict): Dictionary with training, validation, and test datasets.
        model (dict): Dictionary with the following contents.
            'input_var': Theano input variable.
            'target_var': Theano target variable.
            'network': output layer of Lasagne network.
            'loss': Lasagne objective function.
            'updates': optimizer from Lasagne.updates.
        num_epochs (int): Number of epochs to train for.
    """
    print("Compiling model...")
    network, train_fn, val_fn = compile_model(model)

    print("Starting training...")
    print("\n{:>5} {:>17} {:>17} {:>10} {:>10}".format(
        'Epoch', 'Training loss', 'Validation loss', 'Accuracy', 'Time (s)'))
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(inputs['training']['data'],
                                         inputs['training']['labels'],
                                         1,
                                         shuffle=True):
            batch_inputs, batch_targets = batch
            train_err += train_fn(batch_inputs, batch_targets)
            train_batches += 1

        avg_train_err = train_err / train_batches
        avg_val_err, avg_val_acc = full_pass(inputs['validation'], val_fn)
        print("{:>5} {:>17.5f} {:>17.5f} {:>10.3%} {:>10.3f}".format(
            epoch + 1,
            avg_train_err,
            avg_val_err,
            avg_val_acc,
            time.time() - start_time))

    print("\nFinal results:")
    avg_final_err, avg_final_acc = full_pass(inputs['testing'], val_fn)
    print("Loss: {}".format(avg_final_err))
    print("Accuracy: {:.3%}".format(avg_final_acc))

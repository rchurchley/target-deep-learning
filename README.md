## System requirements

The modules used in this project require Python 3.4 and the following Python libraries:

- [Pillow](https://python-pillow.github.io)
- [Requests](http://docs.python-requests.org/en/latest/)
- [flickrapi](http://stuvel.eu/flickrapi)
- [numpy](http://www.numpy.org)
- [Theano](http://deeplearning.net/software/theano/)
- [Lasagne](http://lasagne.readthedocs.org/en/stable/index.html) 0.1

These can all be installed through `pip`, with the exception of Theano (as Lasagne requires a Theano version more recent than the latest stable release). Please see the [Lasagne documentation](http://lasagne.readthedocs.org/en/stable/user/installation.html) for instructions on how to install the correct version of Theano.

Your neural networks can be trained much faster if Theano uses your GPU. To take advantage of this speedup, you will need the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and drivers installed. Theano will use the GPU when `python` is run with the correct flags; you may wish to create an alias to simplify the invocation:

```shell
alias gpupython3="THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python3"
```

To download Flickr images with the `Flickr_Manager` class, you will need a Flickr API key saved in a file called `api_flickr.txt`. This file should consist of two lines: your API key and your API secret.


## Usage

Our experiments consist of three basic steps:

1. Obtain some images.
2. Create a dataset from the images.
3. Train a neural network with the dataset.

They are illustrated by the included sample network and datasets.

```shell
python3 artificial.py
python3 mkdata.py images/black/raw images/black/square data/black+square
gpupython3 runexperiment.py data/black+square experiments/test 100
```

The script `artificial.py` creates four batches of images: solid black backgrounds, solid black backgrounds with white squares, solid coloured backgrounds, and solid coloured backgrounds with white squares. 

Less boring experiments can be run on larger datasets from Flickr. The script `dlflickr.py` downloads up to 5000 images from each of 42 of the most common tags on Flickr, and `mdimage.py` resizes and randomly adds white squares to them.

```shell
python3 dlflickr.py
python3 mdimage.py images/flickr
python3 mkdata.py images/flickr/64 images/flickr/square data/flickr+square
gpupython3 runexperiment.py data/flickr+square experiments/test2 100
```

All of `dlflickr.py`, `mdimage.py`, `mkdata.py`, and `runexperiment.py` are simple illustrations of how to use the `deepsix` library. They can be easily modified to create other datasets and to test deeper neural network structures.

import os
import json
import random
import numpy
from PIL import Image

image_extensions = set(['.jpg', '.jpeg', '.bmp', '.png'])


def image_filenames_as_dict(input_directory):
    """Return {"filename": "/path/to/filename.ext"} for images in a folder."""
    result = {}
    for filename in os.listdir(input_directory):
        root, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            result[root] = os.path.join(input_directory, filename)
    return result


class Dataset:
    """A class for converting folders of images to numpy arrays with labels.

    Attributes:
        directory: A directory path for for the output files.
        sources: A list of paths to directories containing images to be used.
        images: A list of Image_Data each storing the path, source index, and
            a numpy array of the image data.
        parts: A dictionary storing the number of images reserved for the
            training, validation, and testing data sets.
    """

    def __init__(self, directory, sources):
        """Initialize the dataset with image paths.

        Args:
            directory: A directory path for for the output files.
            sources: A list of directory paths containing images.
        """
        os.makedirs(directory, exist_ok=True)  # Ensure `directory` exists
        self.directory = directory
        self.sources = sources
        self.load_paths()
        self.repartition()
        n = len(self.images)
        self.parts = {'training': slice(2*(n//10), n),
                      'validation': slice(n//10, 2*(n//10)),
                      'testing': slice(n//10)}

    def __str__(self):
        """Return a summary of the dataset."""
        result = '{} images from {} sources'.format(len(self.images),
                                                    len(self.sources))
        for purpose in ['training', 'validation', 'testing']:
            dist = []
            images = self.images[self.parts[purpose]]
            for source in range(len(self.sources)):
                dist.append(sum(1 for x in images if x.label == source))
            result += '\n | {:6} for {} {}'.format(sum(dist), purpose, dist)
        return result

    def load_paths(self):
        """Add Image_Data to self.images for each image in self.sources.

        If two sources contain an image with the same id, choose one source
        uniformly at random.
        """
        # find image paths from each source
        path_dictionary = {}
        label = 0
        for source in self.sources:
            for uid, path in image_filenames_as_dict(source).items():
                if uid not in path_dictionary:
                    path_dictionary[uid] = [(path, label)]
                else:
                    path_dictionary[uid].append((path, label))
            label += 1
        # roll a die to decide which source to use for each image id
        for uid, paths in path_dictionary.items():
            source = random.randrange(len(paths))
            path_dictionary[uid] = paths[source]
        # add (path, label) pairs to self.paths
        self.images = []
        for item in path_dictionary.values():
            self.images.append(self.Image_Data(path=item[0],
                                               label=item[1]))

    def repartition(self):
        """Redistribute images among the training/validation/testing sets."""
        random.shuffle(self.images)

    def load_images(self):
        """Load image data from paths for all images in self.images."""
        correct_shape = None
        for image in self.images:
            image.load()
            if correct_shape:
                assert image.image.shape == correct_shape
            correct_shape = image.image.shape

    def save(self):
        """Save the dataset in .npy and JSON files.

        Six .npy files are produced, storing the image data and labels of the
        training, validation, and testing sets, respectively. One JSON file
        stores the list of image paths used in the dataset for better
        reproducibility.
        """
        save_data = {}
        save_data['sources'] = self.sources
        save_data['paths'] = [x.path for x in self.images]
        for part, slice in self.parts.items():
            # load data into numpy array
            data = numpy.array([x.image for x in self.images[slice]])
            labels = numpy.array([x.label for x in self.images[slice]])
            # cast numpy arrays to the appropriate types
            data = data.astype(numpy.float32)
            labels = labels.astype(numpy.int32)
            # save data and labels to .npy files
            data_file = os.path.join(self.directory, part + '_data.npy')
            numpy.save(data_file, data)
            print('Saved array {!s:22} > {}'.format(data.shape, data_file))
            labels_file = os.path.join(self.directory, part + '_labels.npy')
            numpy.save(labels_file, labels)
            print('Saved array {!s:22} > {}'.format(labels.shape, labels_file))
        filename = os.path.join(self.directory, 'datasets.json')
        with open(filename, 'w') as f:
            json.dump(save_data, f)

    class Image_Data:
        """The local path, source label, and pixel data of an image file.

        Attributes:
            path: A string storing the local path of the image.
            label: An integer indicating the source of the image.
            image: A 3D numpy array storing the pixel data of the image.
        """

        def __init__(self, path, label):
            """Initialize image path and label."""
            self.path = path
            self.label = label
            self.image = []

        def load(self):
            """Load RGB image data from the file at self.path to a numpy array.

            The numpy array self.image has shape (3, width, height) with
            subpixel values normalized to the interval [0,1].
            """
            image = numpy.array(Image.open(self.path).convert('RGB'))
            image = numpy.swapaxes(image, 2, 0)
            image = image / 255.
            self.image = image

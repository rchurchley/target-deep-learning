import os
import numpy


def image_filenames_as_dict(input_directory):
    """Return {"filename": "/path/to/filename.jpg"} for images in a directory.

    Extensions are not case-sensitive and include: jpg, jpeg, bmp, png.
    """
    result = {}
    for filename in os.listdir(input_directory):
        root, ext = os.path.splitext(filename)
        if ext.lower() in set(['.jpg', '.jpeg', '.bmp', '.png']):
            result[root] = '{}/{}'.format(input_directory, filename)
    return result

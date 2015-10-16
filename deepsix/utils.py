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


def alter_images(procedure,
                 args,
                 input_directory,
                 output_directory,
                 output_format='JPEG'):
    """Alter all images in an input directory according to some procedure.

    Output files will be saved with the same name as the input files, but
    copied to a different directory.

    Args:
        procedure (function: void): A function that saves an altered image to
            a given path. The function should have keyword arguments args,
            path, output_path, output_format. Some functions of this form
            can be found in deepsix.anomalies.
        args (list): A list of arguments to pass to procedure.
        input_directory (str): A folder containing images to be altered.
        output_directory (str): An existing folder to save images to.
        output_format (str): The image format to save to, e.g. 'JPEG', 'BMP'.
    """
    filename_dict = image_filenames_as_dict(input_directory)
    ensure_directory(output_directory)
    i = 0
    n = len(filename_dict)
    for root, path in filename_dict.iteritems():
        i += 1
        print '{}/{}: Applying {} to {}'.format(i,
                                                n,
                                                procedure.__name__,
                                                root)

        input_path = path
        output_path = '{}/{}.{}'.format(output_directory,
                                        root,
                                        output_format.lower())

        procedure(args=args,
                  path=input_path,
                  output_path=output_path,
                  output_format=output_format)

import os

__all__ = ['images', 'data', 'experiment']

image_extensions = set(['.jpg', '.jpeg', '.bmp', '.png'])


def image_filenames_as_dict(input_directory):
    """Return {"filename": "/path/to/filename.ext"} for images in a folder."""
    result = {}
    for filename in os.listdir(input_directory):
        root, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            result[root] = os.path.join(input_directory, filename)
    return result

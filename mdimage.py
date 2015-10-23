from PIL import Image, ImageDraw
from numpy.random import randint as randint
from deepsix.images import *


def add_square(img):
    """Return the input PIL.Image with a 16x16 white square drawn on it."""
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    size = 16
    x, y = randint(0, img.size[0] - size), randint(0, img.size[1] - size)
    square = [x, y, x + size, y + size]
    draw.rectangle(square, fill=(255, 255, 255))
    del draw
    return img

images = Image_Manager(directory='images/flickr')
images.resize_raws(64)
images.make_versions('square', add_square)
images.save()

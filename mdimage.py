import sys
from PIL import Image, ImageDraw
from random import randrange
from deepsix.images import Image_Manager


def add_square(img):
    """Return the input PIL.Image with a 16x16 white square drawn on it."""
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    size = 16
    x, y = randrange(img.size[0] - size), randrange(img.size[1] - size)
    square = [x, y, x + size, y + size]
    draw.rectangle(square, fill=(255, 255, 255))
    del draw
    return img

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 mdimage.py image/resource/directory')
        exit()
    images = Image_Manager(directory=sys.argv[1])
    images.resize_raws(64)
    images.make_versions('square', add_square)
    images.save()

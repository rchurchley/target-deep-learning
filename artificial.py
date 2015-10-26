import os
from random import randrange
from PIL import Image, ImageDraw

number = 1000
directory = 'images/black'
if not os.path.exists(os.path.join(directory, 'raw')):
    os.makedirs(os.path.join(directory, 'raw'))
if not os.path.exists(os.path.join(directory, 'square')):
    os.makedirs(os.path.join(directory, 'square'))

print('Creating black-and-white images...')
for i in range(1, number + 1):
    img = Image.new('RGB', [64, 64], color=(0, 0, 0))
    img.save(os.path.join(directory, 'raw', '{}.bmp'.format(i)), 'BMP')
    draw = ImageDraw.Draw(img)
    size = 16
    x, y = randrange(img.size[0] - size), randrange(img.size[1] - size)
    square = [x, y, x + size, y + size]
    draw.rectangle(square, fill=(255, 255, 255))
    del draw
    img.save(os.path.join(directory, 'square', '{}.bmp'.format(i)), 'BMP')

print('Creating solid colour background images...')
directory = 'images/solid'
if not os.path.exists(os.path.join(directory, 'raw')):
    os.makedirs(os.path.join(directory, 'raw'))
if not os.path.exists(os.path.join(directory, 'square')):
    os.makedirs(os.path.join(directory, 'square'))
for i in range(number + 1, 2 * number + 1):
    random = (randrange(255), randrange(255), randrange(255))
    img = Image.new('RGB', [64, 64], color=random)
    img.save(os.path.join(directory, 'raw', '{}.bmp'.format(i)), 'BMP')
    draw = ImageDraw.Draw(img)
    size = 16
    x, y = randrange(img.size[0] - size), randrange(img.size[1] - size)
    square = [x, y, x + size, y + size]
    draw.rectangle(square, fill=(255, 255, 255))
    del draw
    img.save(os.path.join(directory, 'square', '{}.bmp'.format(i)), 'BMP')
print('Done.')

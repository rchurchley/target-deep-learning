import os
from random import randrange
from PIL import Image, ImageDraw

number = 1000
directory = 'images/black'
os.makedirs(os.path.join(directory, 'raw'), exist_ok=True)
os.makedirs(os.path.join(directory, 'square'), exist_ok=True)

print('Creating black-and-white images...', end=' ', flush=True)
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
print('done.')

print('Creating solid colour background images...', end=' ', flush=True)
directory = 'images/solid'
os.makedirs(os.path.join(directory, 'raw'), exist_ok=True)
os.makedirs(os.path.join(directory, 'square'), exist_ok=True)
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
print('done.')

import deepsix
from deepsix import data

a = deepsix.data.Dataset('data/example',
                         ['images/flickr/64', 'images/flickr/square'])
print(str(a) + '\n')
a.load_images()
a.save()

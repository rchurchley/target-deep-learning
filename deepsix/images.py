import os
import requests
import shutil
import json
from PIL import Image
import flickrapi
from . import image_filenames_as_dict

requests.packages.urllib3.disable_warnings()


class Image_Manager:
    """A class for collecting, downloading, and modifying images.

    Subclasses may override find_resources() to return results from a
    particular source of images (e.g. Flickr).

    Attributes:
        directory: A directory path for images and related files.
        resources: A set of Image_Resource objects.
    """

    def __init__(self, directory):
        """Initialize image resources."""
        os.makedirs(directory, exist_ok=True)  # Ensure `directory` exists
        self.directory = directory
        self.resources = set()
        # Load existing resources from JSON if possible
        filename = os.path.join(directory, 'resources.json')
        if os.path.exists(filename):
            with open(filename) as f:
                existing_resources = json.load(f)
                for key, paths in existing_resources.items():
                    url = paths[0]
                    default_raw = os.path.join(directory, 'raw', key + '.jpeg')
                    if os.path.exists(paths[1]):
                        raw = paths[1]
                    elif os.path.exists(default_raw):
                        raw = default_raw
                    else:
                        raw = ''
                    self.resources.add(self.Image_Resource(id=key,
                                                           url=url,
                                                           raw=raw))
        # Add resources in directory/raw missing from the JSON for some reason
        raw_dir = os.path.join(directory, 'raw')
        if os.path.exists(raw_dir):
            for uid, path in image_filenames_as_dict(raw_dir).items():
                self.resources.add(self.Image_Resource(id=uid,
                                                       url='',
                                                       raw=path))

    def __str__(self):
        """Return a string representation of the image resource set."""
        return '\n'.join(str(s) for s in self.resources)

    def save(self):
        """Save image resources to a JSON file."""
        filename = os.path.join(self.directory, 'resources.json')
        with open(filename, 'w') as f:
            json.dump({r.id: [r.url, r.raw] for r in self.resources}, f)

    def find_resources(self, **kwargs):
        """Return an iterator of Image_Resources from a source."""
        return []

    def add_resources(self, maximum, **kwargs):
        """Collect a bounded number of Image_Resources from a source."""
        i = 0
        for r in self.find_resources(**kwargs):
            self.resources.add(r)
            i += 1
            if i >= maximum:
                break

    def download_all(self):
        """Download all image resources to self.directory/raw."""
        subdirectory_path = os.path.join(self.directory, 'raw')
        os.makedirs(subdirectory_path, exist_ok=True)
        i, n = 1, len(self.resources)
        invalid = set()
        for r in self.resources:
            try:
                r.download(subdirectory_path)
                print('{}/{}: New file {} sucessfully downloaded.'
                      ''.format(i, n, r.id))
            except RuntimeWarning:
                print('{}/{}: Old file {} already exists.'
                      ''.format(i, n, r.id))
            except ValueError:
                print('{}/{}: Invalid response for {}.'.format(i, n, r.id))
                invalid.add(r)
            i += 1
        self.resources.difference_update(invalid)

    def make_versions(self, version_key, alteration, update_raw=False):
        """Create a new, altered version of each image resource."""
        subdirectory_path = os.path.join(self.directory, version_key)
        os.makedirs(subdirectory_path, exist_ok=True)
        i, n = 1, len(self.resources)
        for r in self.resources:
            try:
                r.make_version(subdirectory_path, alteration, update_raw)
                print('{}/{}: New version of {} sucessfully created.'
                      ''.format(i, n, r.id))
            except RuntimeWarning:
                print('{}/{}: Old version of {} already exists.'
                      ''.format(i, n, r.id))
            i += 1

    def resize_raws(self, size):
        """Resize each image resource and treat the updated image as raw."""
        self.make_versions(
            version_key=str(size),
            alteration=lambda img: img.resize(size=(size, size)),
            update_raw=True)

    class Image_Resource:
        """The URL and local path to a raw image resource.

        Attributes:
            id: A unique string identifying the resource.
            url: A url pointing to a downloadable copy of the image.
            raw: The path to a local "raw" copy of the image.
        """

        def __init__(self, **kwargs):
            """Initialize image resource."""
            self.id = kwargs['id']
            self.url = kwargs['url'] if 'url' in kwargs else ''
            self.raw = kwargs['raw'] if 'raw' in kwargs else ''

        def __str__(self):
            """Return a string representation of the resource."""
            return '{} | {}'.format(self.id, self.raw)

        def __hash__(self):
            """Return a hash of the image resource."""
            return hash(self.id)

        def __eq__(self, other):
            """Return whether two image resources have the same id."""
            return self.id == other.id

        def __ne__(self, other):
            """Return whether two image resources have different ids."""
            return self.id != other.id

        def download(self, directory):
            """Download a raw version of the image to a directory."""
            filename = '{}/{}.jpeg'.format(directory, self.id)
            if self.raw and self.raw == filename:
                raise RuntimeWarning('Already downloaded.')
            else:
                if os.path.exists(filename):
                    self.raw = filename
                    raise RuntimeWarning('A file already exists here.')
                else:
                    self.raw = ''
                    r = requests.get(self.url, stream=True)
                    if all([r.status_code == 200,
                            r.headers['Content-Type'] == 'image/jpeg']):
                        with open(filename, "wb") as out_file:
                            r.raw.decode_content = True
                            shutil.copyfileobj(r.raw, out_file)
                            self.raw = filename
                    else:
                        raise ValueError('Invalid response.')

        def make_version(self, directory, alteration, update_raw=False):
            """Alter the image and save a version in a directory.

            Args:
                directory: The directory path to save the new version in.
                alteration: A function returning a PIL.Image from input img.
                update_raw: If true, the new image will be considered the new
                    raw image.
            """
            filename = '{}/{}.bmp'.format(directory, self.id)
            if os.path.exists(filename):
                raise RuntimeWarning('A version is already here.')
            else:
                img = Image.open(self.raw)
                img = alteration(img)
                img.save(filename, 'BMP')
            if update_raw:
                self.raw = filename


class Flickr_Manager(Image_Manager):
    """A class for collecting, downloading, and modifying Flickr images.

    Attributes:
        directory: A directory path for images and related files.
        resources: A dictionary of Flickr_Resource objects organized by id.
    """

    def __init__(self, api_key, directory):
        """Load a Flickr API key file and initialize image resources."""
        super().__init__(directory)
        with open(api_key) as k:
            api_keys = k.readlines()
            self.__api_key = api_keys[0].rstrip()
            self.__api_secret = api_keys[1].rstrip()

    def find_resources(self, tags):
        """Return an iterator of Flickr_Resources from the Flickr API."""
        session = flickrapi.FlickrAPI(self.__api_key, self.__api_secret)
        for photo in session.walk(tag_mode='all', tags=tags, per_page=500):
            yield self.Flickr_Resource(id=photo.get('id'),
                                       farm=photo.get('farm'),
                                       server=photo.get('server'),
                                       secret=photo.get('secret'))

    class Flickr_Resource(Image_Manager.Image_Resource):
        """The Flickr URL and local path to a raw image resource.

        Attributes:
            id: A unique string identifying the resource.
            url: A url pointing to a downloadable copy of the image.
            raw: The path to a local "raw" copy of the image.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if 'url' not in kwargs:
                template = 'https://farm{}.staticflickr.com/{}/{}_{}.jpg'
                farm = kwargs['farm']
                server = kwargs['server']
                secret = kwargs['secret']
                self.url = template.format(farm, server, self.id, secret)


class Target_Manager(Image_Manager):
    """A class for collecting, downloading, and modifying Target images.

    Attributes:
        directory: A directory path for images and related files.
        resources: A dictionary of Target_Resource objects organized by id.
    """

    def find_resources(self, filename):
        """Return an iterator of Target_Resources from a text file of SKUs."""
        with open(filename) as f:
            for sku in f:
                yield self.Target_Resource(id=sku.strip())

    class Target_Resource(Image_Manager.Image_Resource):
        """The Flickr URL and local path to a raw Target product image.

        Attributes:
            id: A unique string identifying the resource.
            url: A url pointing to a downloadable copy of the image.
            raw: The path to a local "raw" copy of the image.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if 'url' not in kwargs:
                template = ('http://scene7.targetimg1.com/is/image/Target/'
                            '{}?wid={}')
                size = kwargs['size']
                self.url = template.format(self.id, size)

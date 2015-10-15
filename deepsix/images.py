import os
import requests
import shutil
try:
    import simplejson as json
except ImportError:
    import json
import flickrapi

requests.packages.urllib3.disable_warnings()


class Image_Manager:
    """A class for collecting, downloading, and modifying images.

    Subclasses can override find_resources() to return results from a
    particular source of images (e.g. Flickr).

    Attributes:
        resources: A set of Image_Resource objects.
    """

    def __init__(self, existing=None):
        """Initialize image resources from a JSON file."""
        if existing and os.path.exists(existing):
            with open(existing) as json_data:
                resources = json.load(json_data)
                self.resources = set(
                    self.Image_Resource(**r) for r in resources)
        else:
            self.resources = set()

    def __str__(self):
        """Return a string representation of the image resource set."""
        return '\n'.join(str(s) for s in self.resources)

    def save(self, filename):
        """Save image resources to a JSON file."""
        with open(filename, 'w') as f:
            json.dump([r.__dict__ for r in self.resources], f, indent=2)

    def find_resources(self, **kwargs):
        """Return an iterator of Image_Resources from a source."""
        return []

    def add_resources(self, maximum, **kwargs):
        """Collect a bounded number of Image_Resources from a source."""
        i = 0
        for resource in self.find_resources(**kwargs):
            self.resources.add(resource)
            i += 1
            if i >= maximum:
                break

    def download_all(self, directory):
        """Download all raw image resources to a given directory."""
        os.makedirs(directory, exist_ok=True)
        i, n = 1, len(self.resources)
        for r in self.resources:
            try:
                r.download(directory)
                print('{}/{}: Downloaded new file {}'.format(i, n, r.id))
            except RuntimeWarning:
                print('{}/{}: already exists'.format(i, n))
            i += 1

    class Image_Resource:
        """A resource storing an image's url and the paths of local versions.

        Attributes:
            id: A unique string identifying the resource.
            url: A url pointing to a downloadable copy of the image.
            versions: A dictionary storing the paths of various local versions
                of the image. The download() method stores a 'raw' version,
                but altered versions with arbitrary keys can be added.
        """

        def __init__(self, **kwargs):
            """Initialize image resource."""
            self.id = str(kwargs['id'])
            self.url = kwargs['url'] if 'url' in kwargs else None
            self.versions = kwargs['versions'] if 'versions' in kwargs else {}

        def __str__(self):
            """Return a one-line string representation of the resource."""
            return 'Image {} | {}'.format(
                self.id,
                ', '.join(key for key in self.versions))

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
            """Download a 'raw' version of the image to a directory."""
            if 'raw' in self.versions:
                raise RuntimeWarning('Already downloaded')
            else:
                r = requests.get(self.url, stream=True)
                if all([r.status_code == 200,
                        r.headers['Content-Type'] == 'image/jpeg']):
                    filename = '{}/{}.jpeg'.format(directory, self.id)
                    with open(filename, "wb") as out_file:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, out_file)
                    self.versions['raw'] = filename


class Flickr_Manager(Image_Manager):
    """A class for collecting, downloading, and modifying Flickr images.

    Attributes:
        resources: A set of Flickr_Resource objects.
    """

    def __init__(self, api_key, existing=None):
        """Load a Flickr API key file and initialize image resources."""
        super().__init__(existing)
        with open(api_key) as k:
            api_keys = k.readlines()
            self.__api_key = api_keys[0].rstrip()
            self.__api_secret = api_keys[1].rstrip()

    def find_resources(self, tags):
        """Return an iterator of Flickr_Resources from the Flickr API."""
        session = flickrapi.FlickrAPI(self.__api_key, self.__api_secret)
        for photo in session.walk(tag_mode='all', tags=tags):
            yield self.Flickr_Resource(id=photo.get('id'),
                                       farm=photo.get('farm'),
                                       server=photo.get('server'),
                                       secret=photo.get('secret'))

    class Flickr_Resource(Image_Manager.Image_Resource):
        """A resource storing an image's url and the paths of local versions.

        Attributes:
            id: A unique string identifying the resource.
            url: A url pointing to a downloadable copy of the image.
            versions: A dictionary storing the paths of various local versions
                of the image. The download() method stores a 'raw' version,
                but altered versions with arbitrary keys can be added.
        """

        def __init__(self, **kwargs):
            self.id = kwargs['id']
            if 'url' in kwargs:
                self.url = kwargs['url']
            else:
                template = 'https://farm{}.staticflickr.com/{}/{}_{}.jpg'
                farm = kwargs['farm']
                server = kwargs['server']
                secret = kwargs['secret']
                self.url = template.format(farm, server, self.id, secret)
            if 'versions' in kwargs:
                self.versions = kwargs['versions']
            else:
                self.versions = {}


class Target_Manager(Image_Manager):
    """A class for collecting, downloading, and modifying Flickr images.

    Attributes:
        resources: A set of Flickr_Resource objects.
    """

    def find_resources(self, filename):
        """Return an iterator of Target_Resources from a text file of SKUs."""
        with open(filename) as f:
            for sku in f:
                yield self.Target_Resource(id=sku.strip())

    class Target_Resource(Image_Manager.Image_Resource):
        """A resource storing an image's url and the paths of local versions.

        Attributes:
            id: A unique string identifying the resource.
            url: A url pointing to a downloadable copy of the image.
            versions: A dictionary storing the paths of various local versions
                of the image. The download() method stores a 'raw' version,
                but altered versions with arbitrary keys can be added.
        """

        def __init__(self, **kwargs):
            self.id = kwargs['id']
            if 'url' in kwargs:
                self.url = kwargs['url']
            else:
                template = ('http://scene7.targetimg1.com/is/image/Target/'
                            '{}?wid={}')
                size = kwargs['size']
                self.url = template.format(self.id, size)
            if 'versions' in kwargs:
                self.versions = kwargs['versions']
            else:
                self.versions = {}

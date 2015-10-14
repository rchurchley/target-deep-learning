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

    def __init__(self, existing=None):
        if existing and os.path.exists(existing):
            with open(existing) as json_data:
                resources = json.load(json_data)
                self.resources = set(
                    self.Image_Resource(**r) for r in resources)
        else:
            self.resources = set()

    def __str__(self):
        return '\n'.join(str(s) for s in self.resources)

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump([r.__dict__ for r in self.resources], f, indent=2)

    def find_resources(self):
        return []

    def add_resources(self, maximum, **kwargs):
        i = 0
        for resource in self.find_resources(**kwargs):
            self.resources.add(resource)
            i += 1
            if i >= maximum:
                break

    def download_all(self, directory):
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
        def __init__(self, **kwargs):
            self.id = str(kwargs['id'])
            self.url = kwargs['url'] if 'url' in kwargs else None
            self.versions = kwargs['versions'] if 'versions' in kwargs else {}

        def __str__(self):
            return 'Image {}\n | URL: {}\n | Versions: {}'.format(
                self.id,
                self.url,
                self.versions)

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other):
            return self.id == other.id

        def __ne__(self, other):
            return self.id != other.id

        def download(self, directory):
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

    def __init__(self, api_key, existing=None):
        super().__init__(existing)
        with open(api_key) as k:
            api_keys = k.readlines()
            self.api_key = api_keys[0].rstrip()
            self.api_secret = api_keys[1].rstrip()

    def find_resources(self, tags):
        session = flickrapi.FlickrAPI(self.api_key, self.api_secret)
        for photo in session.walk(tag_mode='all', tags=tags):
            yield self.Flickr_Resource(id=photo.get('id'),
                                       farm=photo.get('farm'),
                                       server=photo.get('server'),
                                       secret=photo.get('secret'))

    class Flickr_Resource(Image_Manager.Image_Resource):
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

    def find_resources(self, filename):
        with open(filename) as f:
            for sku in f:
                yield self.Target_Resource(id=sku.strip())

    class Target_Resource(Image_Manager.Image_Resource):
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

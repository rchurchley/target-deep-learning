from deepsix.images import Flickr_Manager
import time

flickr = Flickr_Manager(api_key='api_flickr.txt',
                        directory='images/flickr')
tags = ['nikon', 'canon', 'iphone', 'raw', 'photography', 'photos', 'photo',
        'animals', 'architecture', 'baby', 'beach', 'cat', 'city', 'day',
        'family', 'festival', 'flower', 'food', 'landscape', 'nature', 'park',
        'party', 'people', 'sky', 'snow', 'street', 'sunset', 'travel',
        'trees', 'vacation', 'water', 'winter', 'spring', 'summer', 'autumn',
        'australia', 'canada',  'china', 'europe', 'india', 'japan', 'usa']
for tag in tags:
    print('Finding images tagged \'{}\'... '.format(tag), end='', flush=True)
    start_time = time.time()
    flickr.add_resources(maximum=5000, tags=tag)
    run_time = time.time() - start_time
    print('{}s'.format(round(run_time, 2)))
flickr.download_all()
flickr.save()

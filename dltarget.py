from deepsix.images import Target_Manager
import time

flickr = Target_Manager(directory='images/target')
flickr.add_resources(maximum=10000, filename='api_target_skus.txt', size=64)
flickr.download_all()
flickr.save()

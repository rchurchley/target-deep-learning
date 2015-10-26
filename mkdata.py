import sys
from deepsix.data import Dataset

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python3 path/to/dir1 path/to/dir2 path/to/output')
        exit()
    a = Dataset(sys.argv[3], [sys.argv[1], sys.argv[2]])
    print(str(a) + '\n')
    a.load_images()
    a.save()

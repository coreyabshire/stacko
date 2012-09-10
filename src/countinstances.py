import competition_utilities as cu
import os
import sys
import csv

def countinstances(filename):
    count = 0
    with open(os.path.join(cu.data_path, filename), 'rb') as infile:
        reader = csv.reader(infile)
        header = reader.next()
        for line in reader:
            count += 1
    return count

if __name__ == '__main__':
    print countinstances(sys.argv[1])


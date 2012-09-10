import competition_utilities as cu
import os
import csv
import sys
from random import Random

def quick_sample(file_name, out_name, val):
    r = Random()
    count = 0
    countdown = r.randint(0, val)
    with open(os.path.join(cu.data_path, file_name), 'rb') as infile:
        reader = csv.reader(infile)
        with open(os.path.join(cu.data_path, out_name), 'wb') as outfile:
            writer = csv.writer(outfile)
            header = reader.next()
            writer.writerow(header)
            for line in reader:
                d = line[1] # post creation date
                if countdown == 0:
                    writer.writerow(line)
                    countdown = r.randint(0, 10)
                else:
                    countdown -= 1

if __name__ == '__main__':
    quick_sample(sys.argv[1], sys.argv[2], int(sys.argv[3]))


import competition_utilities as cu
import os
import csv
import sys
from random import Random

def head_sample(file_name, out_name, max_count):
    r = Random()
    count = 0
    with open(os.path.join(cu.data_path, file_name), 'rb') as infile:
        reader = csv.reader(infile)
        with open(os.path.join(cu.data_path, out_name), 'wb') as outfile:
            writer = csv.writer(outfile)
            header = reader.next()
            writer.writerow(header)
            for line in reader:
                d = line[1] # post creation date
                if count < max_count:
                    count += 1
                    writer.writerow(line)
                else:
                    break

if __name__ == '__main__':
    head_sample(sys.argv[1], sys.argv[2], int(sys.argv[3]))


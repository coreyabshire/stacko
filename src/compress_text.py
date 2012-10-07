import competition_utilities as cu
import os
import csv
import sys
from random import Random

def compress_text(file_name, out_name):
    with open(os.path.join(cu.data_path, file_name), 'rb') as infile:
        reader = csv.reader(infile)
        with open(os.path.join(cu.data_path, out_name), 'wb') as outfile:
            writer = csv.writer(outfile)
            header = reader.next()
            writer.writerow(header)
            to_compress = [header.index(name) for name in 'Title,BodyMarkdown'.split(',')]
            for line in reader:
                for i in to_compress:
                    line[i] = len(line[i])
                writer.writerow(line)

if __name__ == '__main__':
    compress_text(sys.argv[1], sys.argv[2])


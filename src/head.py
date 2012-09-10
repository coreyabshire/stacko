import competition_utilities as cu
import os
import sys
import csv

def head(filename, maxcount=10):
    count = 0
    with open(os.path.join(cu.data_path, filename), 'rb') as infile:
        reader = csv.reader(infile)
        header = reader.next()
        for line in reader:
            print line
            count += 1
            if count > maxcount:
                break
    return count

if __name__ == '__main__':
    print head(sys.argv[1], sys.argv[2])


import math
import sys
import os
import csv
import competition_utilities as cu

actual_file = "public_leaderboard_actual.csv"
status_column = {
    'not a real question': 0,
    'not constructive':    1,
    'off topic':           2,
    'open':                3,
    'too localized':       4}

def main():
    submission_file = sys.argv[1]
    with open(os.path.join(cu.data_path, actual_file), 'rb') as actualin:
        adata = csv.reader(actualin)
        header = adata.next()
        with open(os.path.join(cu.data_path, submission_file), 'rb') as subin:
            sdata = csv.reader(subin)
            n = 0
            s = 0
            try:
                while True:
                    arow = adata.next()
                    srow = sdata.next()
                    status = arow[14]
                    label = status_column[status]
                    p = max(min(float(srow[label]), 1-10e-15), 10e-15)
                    #print status, p, math.log(p)
                    s += math.log(p)
                    n += 1
            except:
                pass # finished reading file (TODO: stoperror?
            print n, s
            print ((-1.0 / float(n)) * s)

if __name__ == '__main__':
    main()
    

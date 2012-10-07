import math
import sys
import os
import csv
import competition_utilities as cu

actual_file = "public_leaderboard_actual.csv"
labels = [
    'not a real question',
    'not constructive',
    'off topic',
    'open',
    'too localized'
    ]
status_column = dict((labels[i], i) for i in range(len(labels)))

def main():
    submission_file = sys.argv[1]
    with open(os.path.join(cu.data_path, actual_file), 'rb') as actualin:
        adata = csv.reader(actualin)
        header = adata.next()
        with open(os.path.join(cu.data_path, submission_file), 'rb') as subin:
            sdata = csv.reader(subin)
            n = 0
            s = 0
            correct = 0
            n_cat = {}
            c_cat = {}
            g_cat = {}
            for status in status_column.keys():
                n_cat[status] = 0
                c_cat[status] = 0
                g_cat[status] = 0
            try:
                while True:
                    arow = adata.next()
                    srow = sdata.next()
                    status = arow[14]
                    label = status_column[status]
                    p = max(min(float(srow[label]), 1-10e-15), 10e-15)
                    guessix = srow.index(max(srow))
                    if label == guessix:
                        correct += 1
                        c_cat[status] += 1
                    g_cat[labels[guessix]] += 1
                    #print status, p, math.log(p)
                    s += math.log(p)
                    n += 1
                    n_cat[status] += 1
            except:
                pass # finished reading file (TODO: stoperror?
            print 'Prediction count:    %d' % n
            print 'Sum of all logs:     %d' % s
            print 'Total # correct:     %d' % correct
            print 'Total %% correct:     %4.2f%%' % (float(correct) / float(n) * 100.0)
            print 'Correct, count, and % correct by status'
            for status in labels:
                print '%25s %7d %7d %7d %6.2f%%' % (status, g_cat[status], c_cat[status], n_cat[status], (float(c_cat[status]) / float(n_cat[status]) * 100.0))
            print 'Multiclass Log Loss: %6.4f' % ((-1.0 / float(n)) * s)

if __name__ == '__main__':
    main()
    

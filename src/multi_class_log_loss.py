import math
import sys
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
    adata = cu.get_reader(actual_file)
    sdata = cu.get_submission_reader(submission_file)
    n = 0
    s = 0
    try:
        while True:
            arow = adata.next()
            srow = sdata.next()
            s += math.log(float(srow[status_column[arow[14]]]))
            n += 1
    except:
        pass # finished reading file (TODO: stoperror?
    print ((-1.0 / n) * s)

if __name__ == '__main__':
    main()
    

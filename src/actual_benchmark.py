import competition_utilities as cu
import numpy as np

actual_file = 'public_leaderboard_actual.csv'

actual_lookup = {
    'not a real question': [1.0, 0.0, 0.0, 0.0, 0.0],
    'not constructive':    [0.0, 1.0, 0.0, 0.0, 0.0],
    'off topic':           [0.0, 0.0, 1.0, 0.0, 0.0],
    'open':                [0.0, 0.0, 0.0, 1.0, 0.0],
    'too localized':       [0.0, 0.0, 0.0, 0.0, 1.0] }

def main():
    predictions = [actual_lookup[r[14]] for r in cu.get_reader(actual_file)]
    cu.write_submission("actual_benchmark.csv", predictions)

if __name__=="__main__":
    main()

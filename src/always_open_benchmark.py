import competition_utilities as cu
import numpy as np

def main():
    num_samples = len(cu.get_dataframe("public_leaderboard.csv"))
    predictions = [[0.0,0.0,0.0,1.0,0.0] for i in range(num_samples)]
    cu.write_submission("always_open_benchmark.csv", predictions)

if __name__=="__main__":
    main()

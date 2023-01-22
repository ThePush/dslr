import sys
import os
import ml_toolkit as mltk
import pandas as pd
import numpy as np


def main():
    # Get the file name from the command line
    assert len(sys.argv) == 2, 'Usage: python describe.py <filename.csv>'
    filename = sys.argv[1]
    mltk.check_file(filename)

    # Read the data from the file and extract the columns that contains only numeric values and no missing values
    data = pd.read_csv(filename)
    # Specifically drop the column with name 'Hogwarts House' because its empty values return None
    data = mltk.clean_dataset(data, 'Hogwarts House')

    # Compute the statistics for each column
    computations = pd.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'], columns=data.columns)
    for col in data.columns:
        computations[col]['Count'] = mltk.count(data[col])
        computations[col]['Mean'] = mltk.mean(data[col])
        computations[col]['Std'] = mltk.std_var(data[col])
        computations[col]['Min'] = mltk.min(data[col])
        computations[col]['25%'] = mltk.quartile(data[col])[0]
        computations[col]['50%'] = mltk.median(data[col])
        computations[col]['75%'] = mltk.quartile(data[col])[1]
        computations[col]['Max'] = mltk.max(data[col])

    print(computations)


if __name__ == '__main__':
    main()
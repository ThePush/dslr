import sys
import os
import ml_toolkit as mltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Get the dataset from the directory
    filename = 'datasets/dataset_train.csv'
    mltk.check_file(filename)

    # Read the data from the file and extract the columns that contains only numeric values and no missing values
    data = pd.read_csv(filename)
    data = mltk.drop_missing_rows(data)
    data = mltk.drop_column_by_name(data, 'Index')

    # calculate std_var for each course in each house
    deviation = {}
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            continue
        data[col] = mltk.normalize_column(data[col])
        std_var = mltk.std_var(data[col])
        deviation[col] = std_var
        print(f'Standard deviation of {col} is {std_var}')
    print(f'Course with the minimum standard deviation is {min(deviation, key=deviation.get)}')

    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            continue
        plt.hist(data[data['Hogwarts House'] == 'Slytherin'][col], alpha=0.5, label='Slytherin', color='green')
        plt.hist(data[data['Hogwarts House'] == 'Gryffindor'][col], alpha=0.5, label='Gryffindor', color='red')
        plt.hist(data[data['Hogwarts House'] == 'Ravenclaw'][col], alpha=0.5, label='Ravenclaw', color='blue')
        plt.hist(data[data['Hogwarts House'] == 'Hufflepuff'][col], alpha=0.5, label='Hufflepuff', color='yellow')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.title(f'Histogram of {col}')
        plt.show()

if __name__ == '__main__':
    main()
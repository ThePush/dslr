import srcs.ml_toolkit as ml
import pandas as pd
import sys
import matplotlib.pyplot as plt
import itertools


def main():
    filename = 'datasets/dataset_train.csv'
    try:
        ml.check_file(filename)
    except AssertionError as e:
        print(e)
        sys.exit(1)

    df = pd.read_csv(filename)
    df = ml.drop_columns_by_name(df, ['Index'])
    df = ml.drop_missing_rows(df)

    # Selecting only the numeric columns for plotting
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])

    # Creating scatter plot matrix
    num_features = numeric_cols.shape[1]
    fig, axes = plt.subplots(num_features, num_features, figsize=(12, 12))

    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                axes[i, j].hist(numeric_cols.iloc[:, i].dropna(), bins=20, color='b')
            else:
                axes[i, j].scatter(numeric_cols.iloc[:, j].dropna(), numeric_cols.iloc[:, i].dropna(), color=['r', 'g'])
            
            if i == num_features - 1:
                axes[i, j].set_xlabel(numeric_cols.columns[j])
            
            if j == 0:
                axes[i, j].set_ylabel(numeric_cols.columns[i])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

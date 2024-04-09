import sys
import pandas as pd
import numpy as np
import argparse
import srcs.ml_toolkit as ml
from srcs.LogisticRegression import LogisticRegression
import matplotlib
matplotlib.use('TkAgg')

TARGET = 'Hogwarts House'
HOUSES = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
TO_DROP = ['Index', 'First Name', 'Last Name']

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = ml.drop_columns_by_name(
        df, TO_DROP)
    df['Best Hand'] = ml.label_encode_column(df['Best Hand'])
    df['Birthday'] = ml.normalize_dates(df['Birthday'])
    df = ml.normalize_df(df)
    df[TARGET] = df[TARGET].map(HOUSES)
    df = df.fillna(df.mean())

    return df

def main():
    parser = argparse.ArgumentParser(description="Train a logistic regression model on a given dataset.")
    parser.add_argument("--dataset", default='datasets/dataset_train.csv', type=str, help="Path to the training dataset.")
    
    args = parser.parse_args()
    
    filename = args.dataset
    try:
        ml.check_file(filename)
    except AssertionError as e:
        print(e)
        sys.exit(1)

    df = data_preprocessing(pd.read_csv(filename))
    X_train, y_train = ml.split_features_target(df, TARGET)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    # Initialization
    model = LogisticRegression(batch_size=None)
    model.load_train_set(X_train, y_train)

    # Training and plot
    model.fit()
    model.save_thetas()
    model.plot_cost_history()

if __name__ == '__main__':
    main()

import sys
import pandas as pd
import numpy as np
import srcs.ml_toolkit as ml
from srcs.LogisticRegression import LogisticRegression

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
    train = 'datasets/dataset_train.csv'
    try:
        ml.check_file(train)
    except AssertionError as e:
        print(e)
        sys.exit(1)

    df = data_preprocessing(pd.read_csv(train))
    X_train, y_train = ml.split_features_target(df, TARGET)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    # Initialization
    lr = LogisticRegression()
    lr.load_train_set(X_train, y_train)

    # Training and plot
    lr.fit()
    lr.save_thetas()
    lr.plot_cost_history()


if __name__ == '__main__':
    main()

import sys
import pandas as pd
import numpy as np
import srcs.ml_toolkit as ml
from srcs.LogisticRegression import LogisticRegression


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = ml.drop_columns_by_name(
        df, ['Index', 'First Name', 'Last Name'])
    df['Best Hand'] = ml.label_encode_column(df['Best Hand'])
    df['Birthday'] = ml.normalize_dates(df['Birthday'])
    df = ml.normalize_df(df)
    df['Hogwarts House'] = ml.label_encode_column(
        df['Hogwarts House'])
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
    X_train, y_train, X_test, y_test = ml.train_test_split(
        df, 'Hogwarts House', 0.8)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    lr = LogisticRegression(X_train, y_train)
    lr.fit()
    print(lr.accuracy(X_test, y_test))
    print(y_test.shape)
    lr.print_predictions(X_test, y_test)
    lr.plot_cost_history()
    lr.plot_cm(X_test, y_test)


if __name__ == '__main__':
    main()

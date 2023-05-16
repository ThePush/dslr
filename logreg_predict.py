import sys
import pandas as pd
import numpy as np
import srcs.ml_toolkit as ml
from srcs.LogisticRegression import LogisticRegression

TARGET = 'Hogwarts House'
HOUSES = {0: 'Ravenclaw', 1: 'Slytherin', 2: 'Gryffindor', 3: 'Hufflepuff'}
TO_DROP = ['Index', 'First Name', 'Last Name', TARGET]


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = ml.drop_columns_by_name(
        df, TO_DROP)
    df['Best Hand'] = ml.label_encode_column(df['Best Hand'])
    df['Birthday'] = ml.normalize_dates(df['Birthday'])
    df = ml.normalize_df(df)
    df = df.fillna(df.mean())

    return df


def main():
    test = 'datasets/dataset_test.csv'
    thetas = 'thetas/thetas.csv'
    try:
        ml.check_file(test)
        ml.check_file(thetas)
    except AssertionError as e:
        print(e)
        sys.exit(1)

    df = data_preprocessing(pd.read_csv(test))
    X_test = df.to_numpy()

    # Initialization
    model = LogisticRegression()
    model.load_thetas(thetas)
    model.set_classes(list(HOUSES.keys()))

    # Prediction, stats and plot
    y_hat = model.predict(X_test)
    model.print_predictions(X_test, y_hat)
    print(model.accuracy(X_test, y_hat))
    model.plot_cm(X_test, y_hat)

    # Save predictions
    y_hat = pd.DataFrame(y_hat, columns=[TARGET])
    y_hat[TARGET] = y_hat[TARGET].map(HOUSES)
    y_hat.to_csv('houses.csv', index_label='Index')


if __name__ == '__main__':
    main()

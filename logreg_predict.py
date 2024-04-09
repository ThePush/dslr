import sys
import pandas as pd
import numpy as np
import argparse
import srcs.ml_toolkit as ml
from srcs.LogisticRegression import LogisticRegression
import matplotlib
matplotlib.use('TkAgg')

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
    parser = argparse.ArgumentParser(description="Run logistic regression model to predict Hogwarts houses.")
    parser.add_argument("--dataset", default='datasets/dataset_test.csv', type=str, help="Path to the test dataset.")
    parser.add_argument("--thetas", default='thetas/thetas.csv', type=str, help="Path to the thetas file.")

    args = parser.parse_args()
    
    test = args.dataset
    thetas = args.thetas

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

    # Prediction
    y_hat = model.predict(X_test)
    y_hat = pd.DataFrame(y_hat, columns=[TARGET])
    y_hat[TARGET] = y_hat[TARGET].map(HOUSES)
    y_hat.to_csv('houses.csv', index_label='Index')

if __name__ == '__main__':
    main()

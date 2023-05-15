import pandas as pd
import os
import sys
import numpy as np


def drop_missing_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with missing values from a DataFrame by keeping only
        the rows where the mask is False (no missing values)"""
    clean_data = data.copy()
    clean_data = clean_data[~clean_data.isnull().any(axis=1)]
    return clean_data


def drop_non_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Removes columns that do not contain numerical values from a DataFrame"""
    clean_data = data.copy()
    numeric_columns = []
    for col in clean_data.columns:
        try:
            clean_data[col] = pd.to_numeric(clean_data[col])
            numeric_columns.append(col)
        except ValueError:
            pass
    return clean_data[numeric_columns]


def drop_columns_by_name(data: pd.DataFrame, column_names: list) -> pd.DataFrame:
    """Removes columns from a DataFrame by their names"""
    clean_data = data.copy()
    for column_name in column_names:
        if column_name in clean_data.columns:
            clean_data.drop(column_name, axis=1, inplace=True)

    return clean_data


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_missing_rows(data)
    data = drop_non_numeric_columns(data)
    return data


def check_file(filename: str) -> bool:
    assert os.path.exists(filename), sys.exit(
        f'File does not exist: {filename}')
    assert filename.endswith('.csv'), sys.exit(
        f'File is not a CSV file: {filename}')
    assert os.path.getsize(filename) > 0, sys.exit(
        f'File is empty: {filename}')
    return True


def normalize_df(data: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the values of a DataFrame between 0 and 1"""
    normalized_data = data.copy()
    for col in normalized_data.columns:
        if is_numeric(normalized_data[col][0]):
            normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (
                normalized_data[col].max() - normalized_data[col].min())
    return normalized_data


def normalize_column(column: pd.Series) -> pd.Series:
    """Normalizes the values of a column between 0 and 1"""
    normalized_column = column.copy()
    normalized_column = (normalized_column - normalized_column.min()) / (
        normalized_column.max() - normalized_column.min())
    return normalized_column


def count(x) -> int:
    if len(x) == 0:
        return None
    return sum(1 for value in x if is_numeric(value))


def mean(x) -> float:
    if len(x) == 0:
        return None
    sum_ = 0
    count_ = 0
    for value in x:
        if is_numeric(value):
            sum_ += float(value)
            count_ += 1
    return sum_ / count_ if count_ > 0 else None


def median(x) -> float:
    if len(x) == 0:
        return None
    x = [float(value) for value in x if is_numeric(value)]
    x = sorted(x)
    if len(x) == 0:
        return None
    if len(x) % 2 == 0:
        mid = len(x) // 2
        return float((x[mid] + x[mid - 1]) / 2)
    else:
        mid = len(x) // 2
        return float(x[mid])


def quartile(x) -> tuple:
    x = [float(value) for value in x if is_numeric(value)]
    if len(x) < 4:
        return None
    x = sorted(x)
    return (median(x[0:len(x) // 2]), median(x[(len(x) + 1) // 2:]))


def var(x) -> float:
    x = [float(value) for value in x if is_numeric(value)]
    n = len(x)
    if n < 2:
        return None
    mean_ = mean(x)
    sum_squares = sum((value - mean_) ** 2 for value in x)
    return sum_squares / (n - 1) if n > 1 else None


def std_var(x) -> float:
    x = [float(value) for value in x if is_numeric(value)]
    if len(x) < 2:
        return None
    return var(x) ** 0.5


def is_numeric(x) -> bool:
    try:
        return True if not np.isnan(float(x)) else False
    except ValueError:
        return False


def max(x) -> float:
    if len(x) == 0:
        return None
    _max = None
    for value in x:
        if is_numeric(value):
            value = float(value)
            if _max is None or value > _max:
                _max = value
    return _max


def min(x) -> float:
    if len(x) == 0:
        return None
    _min = None
    for value in x:
        if is_numeric(value):
            value = float(value)
            if _min is None or value < _min:
                _min = value
    return _min


def mean_absolute_deviation(column) -> float:
    column = [float(value) for value in column if is_numeric(value)]
    return mean([abs(value - mean(column)) for value in column])


def coefficient_of_variation(column) -> float:
    column = [float(value) for value in column if is_numeric(value)]
    if len(column) < 2:
        return None
    return (std_var(column) / mean(column)) * 100


def train_test_split(data: pd.DataFrame, target_column: str, train_size: float = 0.8) -> tuple:
    """Splits a dataset into a training set and a test set, specifying the y_hat column.
    return: (X_train, X_test, y_train, y_test)"""
    assert train_size > 0 and train_size < 1, sys.exit(
        'Train size must be between 0 and 1')

    data = data.copy()
    data = data.sample(frac=1).reset_index(drop=True)

    train_size = int(len(data) * train_size)

    X_train = data.iloc[:train_size, :].copy()
    X_train = drop_columns_by_name(X_train, [target_column]).copy()
    y_train = data.loc[:train_size-1, target_column].copy()
    X_test = data.iloc[train_size:, :].copy()
    X_test = drop_columns_by_name(X_test, [target_column]).copy()
    y_test = data.loc[train_size:, target_column].copy()

    return X_train, y_train, X_test, y_test


def split_features_target(data: pd.DataFrame, target_column: str) -> tuple:
    """Splits a dataset into features and target, specifying the y_hat column.
    return: (X, y)"""
    X = data.copy()
    y = X.pop(target_column)
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def add_intercept_df(X: pd.DataFrame) -> pd.DataFrame:
    """Adds a column of 1s to the left of a DataFrame"""
    X = X.copy()
    X.insert(0, 'intercept', 1)
    return X


def label_encode_column(column: pd.Series) -> pd.Series:
    '''Encodes a column of strings into integers'''
    column = column.copy()
    labels = column.dropna().unique()
    labels = {label: i for i, label in enumerate(labels)}
    column = column.map(labels)
    return column


# convert date (eg: 2000-03-30) to number of days since 1/1/1970
def normalize_dates(column: pd.Series) -> pd.Series:
    column = column.copy()
    column = pd.to_datetime(column)
    column = column.map(lambda date: (date - pd.Timestamp('1970-01-01')) /
                        pd.Timedelta('1D'))
    return column

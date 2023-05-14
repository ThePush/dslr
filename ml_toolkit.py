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


def drop_column_by_name(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Removes a column from a DataFrame by its name"""
    clean_data = data.copy()
    columns = [col for col in clean_data.columns if col != column_name]
    clean_data = clean_data[columns]
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

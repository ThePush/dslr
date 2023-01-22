import pandas as pd
import os
import sys


def count(x) -> int:
    '''Computes the number of elements in a given list or array x. The method
    returns the number of elements as an integer, otherwise None if x is an empty
    list or array.'''
    if len(x) == 0:
        return None
    return len(x)


def mean(x) -> float:
    '''Computes the mean of a given non-empty list or array x, using a for-loop.
    The method returns the mean as a float, otherwise None if x is an empty list or
    array.'''
    if len(x) == 0:
        return None
    return sum(x) / len(x)


def median(x) -> float:
    '''Computes the median of a given non-empty list or array x. The method
    returns the median as a float, otherwise None if x is an empty list or array.'''
    if len(x) == 0:
        return None
    x = sorted(x)
    if len(x) % 2 == 0:
        return (float(x[int(len(x) / 2)]) + float(x[int(len(x) / 2) - 1])) / 2
    else:
        return float(x[int(len(x) / 2)])


def quartile(x) -> tuple:
    '''Computes the 1st and 3rd quartiles of a given non-empty array x.
    The method returns the quartile as a float, otherwise None if x is an empty list or
    array.'''
    x = sorted(x)
    return ([median(x[1:len(x) // 2]), median(x[len(x) // 2:])])


def var(x) -> float:
    '''Computes the variance of a given non-empty list or array x, using a for-
    loop. The method returns the variance as a float, otherwise None if x is
    an empty list or array.'''
    if len(x) == 0:
        return None
    _mean = mean(x)
    return sum([(i - _mean) ** 2 for i in x]) / len(x)


def std_var(x) -> float:
    '''Computes the standard deviation of a given non-empty list or array x,
    using a for-loop. The method returns the standard deviation as a float, otherwise
    None if x is an empty list or array.'''
    if len(x) == 0:
        return None
    return var(x) ** 0.5


def is_numeric(x) -> bool:
    '''Checks if a given string x is numeric. The method returns True if x is
    numeric, otherwise False.'''
    try:
        float(x)
        return True
    except ValueError:
        return False


def max(x) -> float:
    '''Computes the maximum of a given non-empty list or array x, using a for-
    loop. The method returns the maximum as a float, otherwise None if x is an empty
    list or array.'''
    if len(x) == 0:
        return None
    _max = x[0]
    for i in x:
        if i > _max:
            _max = i
    return _max


def min(x) -> float:
    '''Computes the minimum of a given non-empty list or array x, using a for-
    loop. The method returns the minimum as a float, otherwise None if x is an empty
    list or array.'''
    if len(x) == 0:
        return None
    _min = x[0]
    for i in x:
        if i < _min:
            _min = i
    return _min


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


def clean_dataset(data, column_name):
    data = drop_column_by_name(data, column_name)
    data = drop_missing_rows(data)
    data = drop_non_numeric_columns(data)
    return data


def check_file(filename: str) -> bool:
    assert os.path.exists(filename), sys.exit(f'File does not exist: {filename}')
    assert filename.endswith('.csv'), sys.exit(f'File is not a CSV file: {filename}')
    assert os.path.getsize(filename) > 0, sys.exit(f'File is empty: {filename}')
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
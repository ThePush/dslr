import pandas as pd
from itertools import combinations
import numpy as np

df = pd.read_csv("datasets/dataset_train.csv")
df = df.drop(columns=["Index"])
numeric_df = df.select_dtypes(include=[np.number])

highest_corr = 0
columns_with_highest_corr = ("", "")

for col1, col2 in combinations(numeric_df.columns, 2):
    corr = numeric_df[col1].corr(numeric_df[col2])
    print(f"Correlation between {col1} and {col2}: {corr}")

    if abs(corr) > abs(highest_corr):
        highest_corr = corr
        columns_with_highest_corr = (col1, col2)

print(
    f"\nColumns with the highest correlation: {columns_with_highest_corr[0]} and {columns_with_highest_corr[1]}"
)
print(f"Highest correlation score: {highest_corr}")

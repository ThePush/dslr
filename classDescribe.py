import sys
import ml_toolkit as ml
import pandas as pd
import numpy as np

class Describe:
    def __init__(self, filename, columns_to_drop=[]):
        try:
            ml.check_file(filename)
        except Exception as e:
            print(e)
            sys.exit(1)

        self.filename = filename
        self.df = pd.read_csv(self.filename)
        if columns_to_drop is not None:
            for col in columns_to_drop:
                self.df = ml.drop_column_by_name(self.df, col)
        self.df = ml.drop_non_numeric_columns(self.df)

        self.computations = pd.DataFrame(
            index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'MAD', 'CV'], columns=self.df.columns)

    def compute(self):
        for col in self.df.columns:
            self.computations[col]['Count'] = '{:.6f}'.format(
                ml.count(self.df[col])) if ml.count(self.df[col]) is not None else np.nan
            self.computations[col]['Mean'] = '{:.6f}'.format(
                ml.mean(self.df[col])) if ml.mean(self.df[col]) is not None else np.nan
            self.computations[col]['Std'] = '{:.6f}'.format(
                ml.std_var(self.df[col])) if ml.std_var(self.df[col]) is not None else np.nan
            self.computations[col]['Min'] = '{:.6f}'.format(
                ml.min(self.df[col])) if ml.min(self.df[col]) is not None else np.nan

            quartiles = ml.quartile(self.df[col])
            if quartiles is not None:
                self.computations[col]['25%'] = '{:.6f}'.format(quartiles[0]) if quartiles[0] is not None else np.nan
                self.computations[col]['75%'] = '{:.6f}'.format(quartiles[1]) if quartiles[1] is not None else np.nan

            self.computations[col]['50%'] = '{:.6f}'.format(
                ml.median(self.df[col])) if ml.median(self.df[col]) is not None else np.nan
            self.computations[col]['Max'] = '{:.6f}'.format(
                ml.max(self.df[col])) if ml.max(self.df[col]) is not None else np.nan
            self.computations[col]['MAD'] = '{:.6f}'.format(ml.mean_absolute_deviation(
                self.df[col])) if ml.mean_absolute_deviation(self.df[col]) is not None else np.nan
            self.computations[col]['CV'] = '{:.6f}'.format(ml.coefficient_of_variation(
                self.df[col])) if ml.coefficient_of_variation(self.df[col]) is not None else np.nan

    def __str__(self):
        return str(self.computations)

    def __repr__(self):
        return repr(self.computations)

import textwrap
import itertools
import srcs.ml_toolkit as ml
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 8})


def main():
    filename = 'datasets/dataset_train.csv'
    try:
        ml.check_file(filename)
    except AssertionError as e:
        print(e)
        sys.exit(1)

    df = pd.read_csv(filename)
    df = ml.drop_columns_by_name(df, ['Index'])
    df = ml.drop_missing_rows(df)

    numeric_cols = df.select_dtypes(include=['float64', 'int64'])

    num_features = numeric_cols.shape[1]
    fig, axes = plt.subplots(num_features, num_features,
                             figsize=(13, 13), tight_layout=True)

    house_colors = {
        'Slytherin': 'green',
        'Gryffindor': 'red',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow'
    }

    colors = itertools.cycle(house_colors.values())

    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                for house in house_colors.keys():
                    axes[i, j].hist(numeric_cols.loc[df['Hogwarts House'] == house, numeric_cols.columns[i]].dropna(),
                                    bins=20, alpha=0.5, label=house, color=house_colors[house])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            else:
                for house in house_colors.keys():
                    axes[i, j].scatter(numeric_cols.loc[df['Hogwarts House'] == house, numeric_cols.columns[j]].dropna(),
                                       numeric_cols.loc[df['Hogwarts House'] ==
                                                        house, numeric_cols.columns[i]].dropna(),
                                       color=house_colors[house], s=1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

            if i == num_features - 1:
                column_name_x = numeric_cols.columns[j]
                split_name_x = column_name_x.split()
                axes[i, j].set_xlabel(
                    '\n'.join(split_name_x), multialignment='center')

            if j == 0:
                column_name_y = numeric_cols.columns[i]
                split_name_y = textwrap.wrap(column_name_y, width=10)
                axes[i, j].set_ylabel(
                    '\n'.join(split_name_y), rotation=90, multialignment='center')
    plt.legend(df['Hogwarts House'].unique(), loc='center left',
               frameon=False, bbox_to_anchor=(1, 0.5))
    plt.show()


if __name__ == '__main__':
    main()

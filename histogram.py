import srcs.ml_toolkit as ml
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def main():
    filename = 'datasets/dataset_train.csv'
    try:
        ml.check_file(filename)
    except AssertionError as e:
        print(e)
        sys.exit(1)

    df = pd.read_csv(filename)
    df = ml.drop_columns_by_name(df, ['Index'])

    deviation = {}
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue
        df[col] = ml.normalize_column(df[col])
        std_var = ml.std_var(df[col])
        deviation[col] = std_var
        print(f'Standard deviation of {col} is {std_var}')
    print(
        f'Course with the lowest standard deviation is {min(deviation, key=deviation.get)}')

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue
        plt.hist(df[df['Hogwarts House'] == 'Slytherin']
                 [col], alpha=0.5, label='Slytherin', color='green')
        plt.hist(df[df['Hogwarts House'] == 'Gryffindor']
                 [col], alpha=0.5, label='Gryffindor', color='red')
        plt.hist(df[df['Hogwarts House'] == 'Ravenclaw']
                 [col], alpha=0.5, label='Ravenclaw', color='blue')
        plt.hist(df[df['Hogwarts House'] == 'Hufflepuff']
                 [col], alpha=0.5, label='Hufflepuff', color='yellow')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.title(f'Histogram of {col}')
        plt.show()


if __name__ == '__main__':
    main()

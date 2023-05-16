import srcs.ml_toolkit as ml
import pandas as pd
import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def main():
    filename = 'datasets/dataset_train.csv'
    try:
        ml.check_file(filename)
    except AssertionError as e:
        print(e)
        sys.exit(1)

    parser = ArgumentParser()
    parser.add_argument('--c1',
                        type=str,
                        default='Astronomy',
                        help='First course to compare')

    parser.add_argument('--c2',
                        type=str,
                        default='Defense Against the Dark Arts',
                        help='Second course to compare')

    args = parser.parse_args()

    df = pd.read_csv(filename)
    df = ml.drop_columns_by_name(df, ['Index'])

    colors = {
        'Slytherin': 'green',
        'Gryffindor': 'red',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow'
    }

    for house, color in colors.items():
        plt.scatter(df[df['Hogwarts House'] == house][args.c1],
                    df[df['Hogwarts House'] == house][args.c2],
                    alpha=0.5, label=house, color=color)

    plt.xlabel(args.c1)
    plt.ylabel(args.c2)
    plt.legend(loc='upper right')
    plt.title(f'{args.c1} vs {args.c2}')
    plt.show()


if __name__ == '__main__':
    main()

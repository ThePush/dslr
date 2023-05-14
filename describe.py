import sys
import ml_toolkit as ml
import pandas as pd
import classDescribe as cd


def main():
    # Get the file name from the command line
    try:
        assert len(sys.argv) == 2, 'Usage: python describe.py <filename.csv>'
        filename = sys.argv[1]
        ml.check_file(filename)
    except AssertionError as e:
        print(e)
        sys.exit(1)

    toDescribe = cd.Describe(filename)
    toDescribe.compute()
    print(toDescribe)


if __name__ == '__main__':
    main()

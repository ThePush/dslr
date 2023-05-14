import sys
import srcs.ml_toolkit as ml
import srcs.Describe as cd


def main():
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

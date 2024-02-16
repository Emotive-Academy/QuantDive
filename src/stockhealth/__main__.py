import sys


def _main() -> None:
    if sys.argv[1:]:
        print("Hello, {}".format(sys.argv[1]))
    else:
        print("Hello, World!")
    return None


if __name__ == "__main__":
    _main()

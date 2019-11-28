import sys

from src.scripts.simulated_sample import hack_script


if __name__ == '__main__':
    start = int(sys.argv[1])
    # start = 0
    hack_script(start)
import sys

from src.scripts.branin_b1 import hack_script
# from src.scripts.main_testing_script import hack_script

if __name__ == '__main__':
    start = int(sys.argv[1])
    # start = 0
    hack_script(start)

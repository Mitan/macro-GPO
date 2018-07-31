import sys

from src.scripts.sim_main_testing_script import hack_script

if __name__ == '__main__':
    start = int(sys.argv[1])
    # start = 66
    hack_script(start)
import sys

from src.scripts.betaTest import hack_script_beta
from src.scripts.main_testing_script import hack_script

if __name__ == '__main__':
    start = int(sys.argv[1])
    # start = 66
    hack_script_beta(start)
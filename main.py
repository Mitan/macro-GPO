import sys

from src.scripts.betaTest import hack_script_beta
# from src.scripts.main_testing_script import hack_script

if __name__ == '__main__':
    start = int(sys.argv[1])
    h = int(sys.argv[2])
    # start = 100
    hack_script_beta(start, h)
    # hack_script(start)
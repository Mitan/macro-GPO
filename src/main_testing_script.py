import sys

from TestScenario import TestScenario

if __name__ == '__main__':

    my_save_folder_root = "../tests/"
    # max horizon
    h_max = 4
    # time steps
    t = 5

    # number of samples per stage
    num_samples = 50

    args = sys.argv

    start = 200
    end = 201

    # for test
    # no command line args => running locally
    assert start < end
    if len(args) == 1:
        for seed in range(start, end):
            TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                         num_samples=num_samples)
    # first argument is seed
    else:
        seed = int(args[1])
        TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                     num_samples=num_samples)

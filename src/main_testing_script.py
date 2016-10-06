import sys

from TestScenario import TestScenario

if __name__ == '__main__':

    my_save_folder_root = "../tests/full/"
    # max horizon
    h_max = 5
    # time steps
    t = 20

    batch_size = 2

    # number of samples per stage
    num_samples = 50

    args = sys.argv

    start = 202
    end = 203

    # for test
    # no command line args => running locally
    assert start < end
    if len(args) == 1:
        for seed in range(start, end):
            TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                         num_samples=num_samples, batch_size=batch_size)
    # first argument is seed
    else:
        seed = int(args[1])
        TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                     num_samples=num_samples, batch_size=batch_size)

import sys

from TestScenario import TestScenario

if __name__ == '__main__':

    my_save_folder_root = "../tests/full/"
    # max horizon
    h_max = 5
    # time steps
    t = 5

    batch_size = 4

    # number of samples per stage
    num_samples = 50

    args = sys.argv

    start = 239
    end = start+1
    assert start < end

    filename = None
    # filename = "./debug_dataset.txt"

    # load dataset locally from file, for debug
    if filename is not None:
        for seed in range(start, end):
            TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                         num_samples=num_samples, batch_size=batch_size, filename=filename)

    # no command line args => running locally with generating datasets
    elif len(args) == 1:
        for seed in range(start, end):
            TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                         num_samples=num_samples, batch_size=batch_size)
    # first argument is seed
    else:
        seed = int(args[1])
        TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                     num_samples=num_samples, batch_size=batch_size)

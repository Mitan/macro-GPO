import os
import sys
import numpy as np

from TestScenario import TestScenario

if __name__ == '__main__':

    # os.system("taskset -p 0xff %d" % os.getpid())



    my_save_folder_root = "../tests/b4_sAD_loc0_h3_x/"
    my_save_folder_root = "../tests/"
    my_save_folder_root = "../testsRoad/44_log_true/"

    filename = '../datasets/slot44/tlog44.dom'

    # max horizon
    h_max = 3
    # time steps
    t = 5

    batch_size = 4

    # number of samples per stage
    # todo note now it is only for anytime
    # for exact algorithms see SampleFunctionBuilder
    num_samples = 200

    args = sys.argv

    start = 21
    end = start + 1
    assert start < end

    #filename = None
    # filename = "./debug_dataset.txt"


    seed_0 = int(args[1])
    for seed in range(seed_0, seed_0+10):
        TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                 num_samples=num_samples, batch_size=batch_size, filename= filename)

    """
    # load dataset locally from file, for debug
    if filename is not None:
        for seed in range(start, end):
            TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                         num_samples=num_samples, batch_size=batch_size, filename=filename)

    # no command line args => running locally with generating datasets

    else:
    #elif len(args) == 1:
        #for seed in range(start, end):
        seed = int(args[1])
        TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                         num_samples=num_samples, batch_size=batch_size)
            # first argument is seed


    else:
        seed = int(args[1])
        # test_iteration = int(args[2])
        # my_save_folder_root = my_save_folder_root + str(test_iteration) + "/"
        # for seed in range(seed, seed+10):
        filename = my_save_folder_root + "seed" + str(seed) + "/dataset.txt"
        print filename
        TestScenario(my_save_folder_root=my_save_folder_root, h_max=h_max, seed=seed, time_steps=t,
                     num_samples=num_samples, batch_size=batch_size, filename=filename)
    """
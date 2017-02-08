import os
import sys
import numpy as np

from TestScenario import TestScenario

if __name__ == '__main__':

    # os.system("taskset -p 0xff %d" % os.getpid())



    my_save_folder_root = "../tests/b4_sAD_loc0_h3_x/"
    my_save_folder_root = "../tests/"
    # my_save_folder_root = "../testsRoad/h5/44_log_true/"

    # filename = '../datasets/slot18/taxi18.dom'

    # max horizon
    h_max = 3

    t, batch_size, num_samples, time_slot = (5,4, 150, 44)

    filename = '../datasets/slot' + str(time_slot) + '/tlog'+ str(time_slot) +'.dom'

    my_save_folder_root = "../testsRoad/b" + str(batch_size) + "/"+ str(time_slot) + "/"


    # number of samples per stage
    # todo note now num_samples is only for anytime
    # for exact algorithms see SampleFunctionBuilder

    args = sys.argv

    #filename = None
    # filename = "./debug_dataset.txt"


    seed_0 = int(args[1])
    for seed in range(seed_0, seed_0+5):
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
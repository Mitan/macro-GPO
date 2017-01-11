import os
import sys
import numpy as np

from TestScenario import TestScenario, TestScenario_Beta

if __name__ == '__main__':

    # os.system("taskset -p 0xff %d" % os.getpid())

    my_save_folder_root = "../testBeta2/"


    # time steps
    t = 5

    h = 2

    batch_size = 4

    args = sys.argv

    beta_list = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]

    # number of samples per stage
    # todo note now it is only for anytime
    # for exact algorithms see SampleFunctionBuilder
    num_samples = 150

    seed = int(args[1])
    # seed = 66
    #test_iteration = int(args[2])
    #my_save_folder_root = my_save_folder_root + str(test_iteration) + "/"
    #for seed in range(seed, seed+10):
    filename = my_save_folder_root + "seed" + str(seed) + "/dataset.txt"
    print filename
    TestScenario_Beta(test_horizon= h, my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                     num_samples=num_samples, batch_size=batch_size, filename=filename, beta_list=beta_list)

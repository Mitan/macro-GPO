import os
import sys
import numpy as np

from TestScenario import TestScenario, TestScenario_Beta

if __name__ == '__main__':

    h = 2
    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]
    beta_list = [10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 5 * 10 ** -3, 10 ** -2, 5 * 10 ** -2, 0.0, 0.1, 1.0,
                 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]

    args = sys.argv
    seed_0 = int(args[1])
    time_slot = int(args[2])

    # seed_0, time_slot = (0, 18)

    t, batch_size, num_samples = (4, 5, 250)

    filename = '../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'

    my_save_folder_root = "../testsRoadBeta2/b" + str(batch_size) + "/" + str(time_slot) + "/"

    for seed in range(seed_0, seed_0 + 2):
        TestScenario_Beta(my_save_folder_root=my_save_folder_root, test_horizon=h, seed=seed, time_steps=t,
                          num_samples=num_samples, batch_size=batch_size, filename=filename, time_slot=time_slot,
                          beta_list=beta_list)

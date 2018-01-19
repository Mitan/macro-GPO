import os
import sys
import numpy as np

from TestScenario import TestScenario, TestScenario_Beta

if __name__ == '__main__':

    my_save_folder_root = "../simulatedBeta2/"
    my_save_folder_root = "../simulated_tests/beta2/"

    t = 5

    h = 2

    batch_size = 4

    args = sys.argv

    num_samples = 100
    # beta_list = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    beta_list = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    # seed = int(args[1])
    seed = 66

    for seed in range(seed, seed + 2):
        filename = my_save_folder_root + "seed" + str(seed) + "/dataset.txt"
        print filename

        TestScenario_Beta(test_horizon=h, my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                          num_samples=num_samples, batch_size=batch_size, filename=filename, beta_list=beta_list)

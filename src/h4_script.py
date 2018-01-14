import sys

from TestScenario import *

if __name__ == '__main__':

    my_save_folder_root = "../simulated_tests/h4_samples/"

    t, batch_size = (5, 4)

    h = 4
    # num_samples = 300
    samples = [5, 20]
    samples = [20]

    args = sys.argv
    start = int(args[1])
    # start = 66
    for num_samples in samples:
        for seed in range(start, start + 2):
            filename = my_save_folder_root + "seed" + str(seed) + "/dataset.txt"
            print seed, num_samples, h
            TestScenario__only_H4(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                                  batch_size=batch_size, filename=filename, h=h, num_samples=num_samples)

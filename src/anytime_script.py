import sys

from TestScenario import *

if __name__ == '__main__':

    my_save_folder_root = "../11anytime/"

    t = 5

    batch_size = 4

    num_samples = 250

    args = sys.argv

    start = int(args[1])
    # start = 66
    for seed in range(start, start + 1):
        filename = my_save_folder_root + "seed" + str(seed) + "/dataset.txt"
        print seed
        # TestScenario_AnytimeMLE4(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
        #                         batch_size=batch_size, filename=filename)

        TestScenario_H4(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                        batch_size=batch_size, filename=filename, num_samples=num_samples, h_max=3)

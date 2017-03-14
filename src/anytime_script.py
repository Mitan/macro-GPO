import sys

from TestScenario import *

if __name__ == '__main__':

    my_save_folder_root = "../anytime/"

    t = 5

    batch_size = 4

    num_samples = 150

    args = sys.argv

    start = int(args[1])

    for seed in range(start, start+1):
        filename = my_save_folder_root + "seed" + str(seed) + "/dataset.txt"
        print seed
        TestScenario_AnytimeMLE4(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                        num_samples=num_samples, batch_size=batch_size, filename=filename)

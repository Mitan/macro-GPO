import sys

from TestScenario import TestScenario_PE

if __name__ == '__main__':

    my_save_folder_root = "../releaseTests/simulated/rewards-sAD/"

    t = 5

    batch_size = 4

    num_samples = 150

    args = sys.argv

    start = 66
    end = 102

    for seed in range(start, end):
        filename = my_save_folder_root + "seed" + str(seed) + "/dataset.txt"
        print seed
        TestScenario_PE(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                        num_samples=num_samples, batch_size=batch_size, filename=filename)

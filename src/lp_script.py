import sys

from TestScenario import *

if __name__ == '__main__':

    my_save_folder_root = "../releaseTests/simulated/rewards-sAD/"
    my_save_folder_root = "../releaseTests/updated_release/simulated/rewards-sAD/"
    #my_save_folder_root = "../simulated_tests/mle/"

    t = 5

    batch_size = 4

    num_samples = 100

    args = sys.argv

    # start = 66
    # start = int(sys.argv[1])

    # end = 102

    for seed in range(72, 102):
        filename = my_save_folder_root + "seed" + str(seed) + "/dataset.txt"
        print seed
        TestScenario_LP(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                             num_samples=num_samples, batch_size=batch_size, filename=filename)

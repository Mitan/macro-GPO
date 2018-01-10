import sys

from TestScenario import *

if __name__ == '__main__':

    args = sys.argv

    seed_0 = int(args[1])

    # note hardcoded
    time_slot = 18
    t, batch_size, num_samples = (4, 5, 300)

    filename = '../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'

    my_save_folder_root = "../road_tests/21testsfull/"

    for seed in range(seed_0, seed_0 + 7):
        # for seed in seeds:
        TestScenario_2Full(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t, num_samples=num_samples,
                           batch_size=batch_size, filename=filename, time_slot=time_slot)

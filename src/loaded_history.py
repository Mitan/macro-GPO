import sys

from TestScenario import *

if __name__ == '__main__':

    args = sys.argv

    seed_0 = int(args[1])
    # seed_0 = 0

    # note hardcoded
    time_slot = 18
    t, batch_size = (4, 5)

    filename = '../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'

    my_save_folder_root = "../road_tests/h4_1step/"

    # num_samples = 300

    samples = [5, 50]
    samples = [300]
    h = 4

    for num_samples in samples:
        for seed in range(seed_0, seed_0 + 1):
            # for seed in seeds:
            TestScenario_H4_LoadedHistory(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                                          num_samples=num_samples,
                                          batch_size=batch_size, filename=filename, time_slot=time_slot, h=h)

import sys

from TestScenario import *

if __name__ == '__main__':

    args = sys.argv

    # seeds = map(int, args[1:])
    seed_0 = int(args[1])
    # seed_0 = 0
    # time_slot = int(args[2])

    # note hardcoded
    time_slot = 18
    t, batch_size, num_samples = (1, 5, 250)

    filename = '../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'

    my_save_folder_root = "../testsRoad2/b" + str(batch_size) + "/" + str(time_slot) + "/"
    my_save_folder_root = "../testsRoad_4/"
    my_save_folder_root = "../testsRoad_4/"
    my_save_folder_root = "../road_tests/new_h4/"

    for seed in range(seed_0, seed_0 + 8):
        # for seed in seeds:
        TestScenario_H4(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t, num_samples=num_samples,
                           batch_size=batch_size, filename=filename, time_slot=time_slot)

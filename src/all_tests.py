import sys

from TestScenario import *

if __name__ == '__main__':

    args = sys.argv

    seed_0 = 4
    # seed_0 = int(args[1])

    time_slot = 16
    # t, batch_size, num_samples = (4, 5, 300)
    t, batch_size = (4, 5)

    my_save_folder_root = "../a_robot/all_tests/"

    data_file = '../datasets/robot/selected_slots/slot_' + str(time_slot) + '/noise_final_slot_' + str(time_slot) + '.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'

    num_samples = 300
    # samples = [5, 50]
    h = 4
    # for num_samples in samples:
    for seed in range(seed_0, seed_0 + 1):
            TestScenario_H4(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t, num_samples=num_samples,
                            batch_size=batch_size, time_slot=time_slot, coords_filename=coords_file,
                            data_filename=data_file, neighbours_filename=neighbours_file, h=h)
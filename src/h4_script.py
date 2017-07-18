import sys

from TestScenario import *

if __name__ == '__main__':

    args = sys.argv

    # seeds = map(int, args[1:])
    seed_0 = int(args[1])
    # seed_0 = 0

    time_slot = 16
    t, batch_size, num_samples = (4, 5, 250)
    # t, batch_size, num_samples = (4, 5, 1)

    my_save_folder_root = "../robot_tests/tests1_" + str(time_slot) + "/"
    data_file = '../datasets/robot/selected_slots/slot_' + str(time_slot) + '/final_slot_' + str(time_slot) + '.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'

    for seed in range(seed_0, seed_0 + 11):
        TestScenario_H4(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t, num_samples=num_samples,
                        batch_size=batch_size, time_slot=time_slot, coords_filename=coords_file,
                        data_filename=data_file, neighbours_filename=neighbours_file)

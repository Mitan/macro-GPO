import sys

from TestScenario import TestScenario_Beta

if __name__ == '__main__':

    args = sys.argv
    seed_0 = int(args[1])
    h = int(args[2])
    beta = float(args[3])

    time_slot = 16

    t, batch_size, num_samples = (4, 5, 300)

    my_save_folder_root = "../robot_tests/beta3" + str(h) + "/"

    data_file = '../datasets/robot/selected_slots/slot_' + str(time_slot) + '/noise_final_slot_' + str(time_slot) + '.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'

    for seed in range(seed_0, seed_0 + 3):
        TestScenario_Beta(my_save_folder_root=my_save_folder_root, test_horizon=h, seed=seed, time_steps=t,
                          num_samples=num_samples, batch_size=batch_size, time_slot=time_slot,
                          beta=beta, coords_filename=coords_file,
                          data_filename=data_file, neighbours_filename=neighbours_file)

from src.TestScenario import *

if __name__ == '__main__':

    seeds = range(35)

    # note hardcoded
    time_slot = 16
    t, batch_size, num_samples = (20, 1, -1)
    """
    filename = '../../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'

    my_save_folder_root = '../../releaseTests/road/tests2full/'
    """
    my_save_folder_root = '../releaseTests/robot/h2_full/'
    my_save_folder_root = '../noise_robot_tests/all_tests/'
    data_file = '../datasets/robot/selected_slots/slot_' + str(time_slot) + '/noise_final_slot_' + str(time_slot) + '.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'

    for seed in seeds:
        print seed
        TestScenario_EI_PI(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t, num_samples=num_samples,
                           batch_size=batch_size,  time_slot=time_slot, coords_filename=coords_file,
                           data_filename=data_file, neighbours_filename=neighbours_file)

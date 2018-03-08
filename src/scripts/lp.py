from src.TestScenario import *

if __name__ == '__main__':

    seeds = range(3,35)

    # note hardcoded
    """
    time_slot = 18
    t, batch_size, num_samples = (4, 5, 250)

    filename = '../../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'

    my_save_folder_root = '../../releaseTests/road/b5-18-log/'
    """
    time_slot = 16
    t, batch_size, num_samples = (4, 5, 300)
    # t, batch_size, num_samples = (4, 5, 1)

    my_save_folder_root = "../robot_tests/tests1_" + str(time_slot) + "_ok/"
    my_save_folder_root = "../noise_robot_tests/all_tests/"
    my_save_folder_root = "../noise_robot_tests/all_tests/"
    my_save_folder_root = '../releaseTests/updated_release/robot/all_tests_release/'

    data_file = '../datasets/robot/selected_slots/slot_' + str(time_slot) + '/noise_final_slot_' + str(time_slot) + '.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'

    for seed in seeds:
        print seed
        TestScenario_LP(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                                 num_samples=num_samples,
                                 batch_size=batch_size, coords_filename=coords_file,
                                 data_filename=data_file, neighbours_filename=neighbours_file, time_slot=time_slot)

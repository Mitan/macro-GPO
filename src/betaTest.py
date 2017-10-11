import sys

from TestScenario import TestScenario_Beta

if __name__ == '__main__':

    h = 2
    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]
    beta_list = [0.0, 0.01, 0.05,0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    # beta_list = [0.0]

    args = sys.argv
    seed_0 = int(args[1])
    # seed_0 = 0
    #seed_0 = 6
    # time_slot = int(args[2])
    time_slot = 16
    # seed_0, time_slot = (0, 18)

    t, batch_size, num_samples = (4, 5, 250)
    # t, batch_size, num_samples = (4, 5, 10)
    """
    filename = '../../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'

    my_save_folder_root = "../testsRoadBeta3/b" + str(batch_size) + "/" + str(time_slot) + "/"
    my_save_folder_root = "../../releaseTests/road/beta2/"
    """
    my_save_folder_root = "../robot_tests/beta" + str(h) +"/"
    data_file = '../datasets/robot/selected_slots/slot_' + str(time_slot) + '/final_slot_' + str(time_slot) + '.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'

    for seed in range(seed_0, seed_0 + 7):
        TestScenario_Beta(my_save_folder_root=my_save_folder_root, test_horizon=h, seed=seed, time_steps=t,
                          num_samples=num_samples, batch_size=batch_size, time_slot=time_slot,
                          beta_list=beta_list, coords_filename=coords_file,
                        data_filename=data_file, neighbours_filename=neighbours_file )

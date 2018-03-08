from src.TestScenario import TestScenario_PE_qEI_BUCB

if __name__ == '__main__':

    seeds = range(35)

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
    my_save_folder_root = '../../robot_tests/21_full/'

    for seed in seeds:
        print seed
        TestScenario_PE_qEI_BUCB(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                                 num_samples=num_samples,
                                 batch_size=batch_size, time_slot=time_slot)

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
    my_save_folder_root = '../robot_tests/21_full/'

    for seed in seeds:
        print seed
        TestScenario_EI_PI(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t, num_samples=num_samples,
                           batch_size=batch_size,  time_slot=time_slot)

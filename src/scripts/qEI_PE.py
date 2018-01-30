from src.TestScenario import TestScenario_PE_qEI_BUCB

if __name__ == '__main__':

    seeds = range(35)
    # seeds = range(42)

    # note hardcoded
    time_slot = 18
    t, batch_size, num_samples = (4, 5, 300)

    filename = '../../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'

    my_save_folder_root = '../../releaseTests/updated_release/road/b5-18-log/'
    #my_save_folder_root = '../../new_road_tests/new_all/'
    # my_save_folder_root = '../../new_road_tests/new_all_3/'

    for seed in seeds:
        print seed
        TestScenario_PE_qEI_BUCB(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t,
                                 num_samples=num_samples,
                                 batch_size=batch_size, filename=filename, time_slot=time_slot)

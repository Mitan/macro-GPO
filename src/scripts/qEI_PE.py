from src.TestScenario import TestScenario_PE_qEI

if __name__ == '__main__':

    seeds = range(10)

    # note hardcoded
    time_slot = 18
    t, batch_size, num_samples = (4, 5, 250)

    filename = '../../datasets/slot' + str(time_slot) + '/tlog' + str(time_slot) + '.dom'

    my_save_folder_root = '../../releaseTests/road/b5-18-log/'

    for seed in seeds:
        print seed
        TestScenario_PE_qEI(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t, num_samples=num_samples,
                            batch_size=batch_size, filename=filename, time_slot=time_slot)

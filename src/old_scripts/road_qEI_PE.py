from src.TestScenario import TestScenario_PE_qEI_BUCB
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum

if __name__ == '__main__':

    seeds = range(35)
    seeds = range(66, 102)

    # note hardcoded
    time_slot = 18
    time_slot = 16
    t, batch_size, num_samples = (4, 5, 250)
    t, batch_size, num_samples = (5, 4, 250)

    my_save_folder_root = '../../robot_tests/road/'
    my_save_folder_root = '../../releaseTests/updated_release/road/b5-18-log/'
    my_save_folder_root = '../../releaseTests/updated_release/robot/all_tests_release/'
    my_save_folder_root = '../../releaseTests/updated_release/simulated/rewards-sAD/'

    for seed in seeds:
        print seed
        TestScenario_PE_qEI_BUCB(my_save_folder_root=my_save_folder_root,
                                 seed=seed, time_steps=t,
                                 num_samples=num_samples,
                                 batch_size=batch_size,
                                 time_slot=time_slot,
                                 dataset_type=DatasetEnum.Simulated,
                                 dataset_mode=DatasetModeEnum.Load,
                                 ma_treshold=20
                                 )

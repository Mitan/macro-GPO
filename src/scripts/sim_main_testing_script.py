import sys

from src.newTestScenario import *
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum


def hack_script(start):
    my_save_folder_root = "../releaseTests/updated_release/simulated/rewards-sAD/"
    my_save_folder_root = "../sim-fixed-temp/"
    my_save_folder_root = "./h4-b1/"

    batch_size = 1

    t = 20 / batch_size

    num_samples = 20
    anytime_num_samples = 300

    end = start + 5
    assert start < end
    for seed in range(start, end):
        print seed
        TestScenario_all_tests(my_save_folder_root=my_save_folder_root,
                               seed=seed,
                               time_steps=t,
                               num_samples=num_samples,
                               anytime_num_samples=anytime_num_samples,
                               batch_size=batch_size,
                               time_slot=18,
                               dataset_type=DatasetEnum.Simulated,
                               dataset_mode=DatasetModeEnum.Load,
                               ma_treshold=20)

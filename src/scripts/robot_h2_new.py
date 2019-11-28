import sys

from src.newTestScenario import *
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum


def hack_script(start):
    batch_size = 5

    total_budget = 20

    num_samples = 300

    my_save_folder_root = "./tests/robot_iter_h2_b{}_s{}/".format(batch_size, num_samples)
    my_save_folder_root = "./tests/1_road_iter_h2_b{}_s{}/".format(batch_size, num_samples)

    end = start + 1
    assert start < end

    for seed in range(start, end):
        TestScenario_h2_robot(my_save_folder_root=my_save_folder_root,
                              seed=seed,
                              total_budget=total_budget,
                              num_samples=num_samples,
                              batch_size=batch_size,
                              time_slot=18,
                              dataset_type=DatasetEnum.Road,
                              dataset_mode=DatasetModeEnum.Generate,
                              ma_treshold=20)

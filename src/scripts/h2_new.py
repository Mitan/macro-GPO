import sys

from src.newTestScenario import *
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum


def hack_script(start):
    batch_size = 5

    total_budget = 20

    num_samples = 300
    anytime_num_iterations = 1500

    # my_save_folder_root = "./tests/branin_400_b4_s100_new/"
    my_save_folder_root = "./tests/camel_600_b{}_s{}/".format(batch_size, num_samples)
    my_save_folder_root = "./tests/gold_400_b{}_s{}/".format(batch_size, num_samples)
    my_save_folder_root = "./tests/boha_400_b{}_s{}/".format(batch_size, num_samples)
    my_save_folder_root = "./tests/road_h2_b{}_s{}/".format(batch_size, num_samples)

    end = start + 20
    assert start < end

    for seed in range(start, end):
        TestScenario_h2(my_save_folder_root=my_save_folder_root,
                        seed=seed,
                        total_budget=total_budget,
                        num_samples=num_samples,
                        batch_size=batch_size,
                        time_slot=18,
                        dataset_type=DatasetEnum.Road,
                        dataset_mode=DatasetModeEnum.Generate,
                        anytime_num_iterations=anytime_num_iterations,
                        ma_treshold=20)

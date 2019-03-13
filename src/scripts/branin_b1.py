import sys

from src.newTestScenario import *
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum


def hack_script(start):
    my_save_folder_root = "./tests/branin-b1-h4/"

    batch_size = 1

    num_samples = 20

    end = start + 2
    assert start < end
    for seed in range(start, end):
        TestScenario_branin(my_save_folder_root=my_save_folder_root,
                            seed=seed,
                            total_budget=20,
                            num_samples=num_samples,
                            batch_size=batch_size,
                            time_slot=18,
                            dataset_type=DatasetEnum.Branin,
                            dataset_mode=DatasetModeEnum.Generate,
                            ma_treshold=20)

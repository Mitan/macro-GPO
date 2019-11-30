import sys

from src.newTestScenario import *
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum


def hack_script(start):
    batch_size = 4

    total_budget = 20

    my_save_folder_root = "./tests/simulated_h4/"

    end = start + 1
    assert start < end

    for seed in range(start, end):
        TestScenario_h3_simulated(my_save_folder_root=my_save_folder_root,
                                  seed=seed,
                                  total_budget=total_budget,
                                  batch_size=batch_size,
                                  time_slot=None,
                                  dataset_type=DatasetEnum.Simulated,
                                  dataset_mode=DatasetModeEnum.Load,
                                  ma_treshold=20)

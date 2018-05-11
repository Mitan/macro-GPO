import sys

from src.newTestScenario import *
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum


def hack_script(start):
    my_save_folder_root = "./sim_4/"
    # max horizon
    h_max = 4
    # time steps

    batch_size = 4

    t = 20 / batch_size

    num_samples = 100
    anytime_num_samples = 300
    # args = sys.argv

    # start = args[1]
    end = start + 1
    assert start < end
    for seed in range(start, end):
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

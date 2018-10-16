from src.newTestScenario import *
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum


def hack_script_beta(start, h):


    batch_size = 4
    total_budget = 20
    # t = 20 / batch_size

    num_samples = 500
    # h = 2

    my_save_folder_root = "./tests/new_beta%d/" % h

    # beta_list = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    beta_list = [0.15]

    end = start + 10
    assert start < end
    for seed in range(start, end):
        print seed
        TestScenario_beta(my_save_folder_root=my_save_folder_root,
                          seed=seed,
                          total_budget=total_budget,
                          num_samples=num_samples,
                          beta_list=beta_list,
                          batch_size=batch_size,
                          time_slot=None,
                          dataset_type=DatasetEnum.Simulated,
                          dataset_mode=DatasetModeEnum.Load,
                          ma_treshold=20,
                          h=h)

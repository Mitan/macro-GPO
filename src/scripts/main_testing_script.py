from src.newTestScenario import *
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum


def hack_script(start):
    # my_save_folder_root = "./tests/sim-fixed-temp/"
    my_save_folder_root = "./tests/road/"

    batch_size = 5
    total_budget = 20

    num_samples = None
    anytime_num_samples = 250
    anytime_num_iterations = 1500
    end = start + 5
    assert start < end
    for seed in range(start, end):
        print seed
        # TestScenario_all_tests(my_save_folder_root=my_save_folder_root,
        TestScenario_all_tests_road(my_save_folder_root=my_save_folder_root,
                                    seed=seed,
                                    total_budget=total_budget,
                                    num_samples=num_samples,
                                    anytime_num_samples=anytime_num_samples,
                                    anytime_num_iterations=anytime_num_iterations,
                                    batch_size=batch_size,
                                    time_slot=18,
                                    dataset_type=DatasetEnum.Road,
                                    dataset_mode=DatasetModeEnum.Generate,
                                    ma_treshold=20)


import sys

from src.TestScenario import TestScenario_PE_qEI_BUCB
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum

if __name__ == '__main__':

    my_save_folder_root = "../../tests4/"
    # max horizon
    h_max = 4
    # time steps
    t = 5

    batch_size = 4

    num_samples = 300

    args = sys.argv

    start = 2
    end = start + 1
    assert start < end
    for seed in range(start, end):
        TestScenario_PE_qEI_BUCB(my_save_folder_root=my_save_folder_root,
                                 seed=seed,
                                 time_steps=t,
                                 num_samples=num_samples,
                                 batch_size=batch_size,
                                 time_slot=16,
                                 dataset_type=DatasetEnum.Robot,
                                 dataset_mode=DatasetModeEnum.Load,
                                 ma_treshold=20)

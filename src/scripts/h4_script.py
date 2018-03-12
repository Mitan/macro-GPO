import sys

from src.TestScenario import *

if __name__ == '__main__':

    args = sys.argv

    # seed_0 = int(args[1])
    seed_0 = 3

    time_slot = 16
    # t, batch_size, num_samples = (4, 5, 300)
    t, batch_size = (4, 5)

    my_save_folder_root = '../../robot_tests/21_full/'

    num_samples = 2
    h = 4
    for seed in range(seed_0, seed_0 + 3):
            TestScenario_H4(my_save_folder_root=my_save_folder_root, seed=seed, time_steps=t, num_samples=num_samples,
                            batch_size=batch_size, time_slot=time_slot, h=h)

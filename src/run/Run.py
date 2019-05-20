import os

from src.TreePlanTester import testWithFixedParameters
from src.dataset_model.DatasetGenerator import DatasetGenerator


def run(batch_size,
        total_budget,
        seeds,
        dataset_type,
        dataset_mode,
        dataset_root_folder,
        results_save_root_folder,
        ma_threshold,
        methods):
    try:
        os.makedirs(results_save_root_folder)
    except OSError:
        if not os.path.isdir(results_save_root_folder):
            raise
    dataset_generator = DatasetGenerator(dataset_type=dataset_type,
                                         dataset_mode=dataset_mode,
                                         batch_size=batch_size,
                                         dataset_root_folder=dataset_root_folder)

    for seed in seeds:
        seed_save_folder = results_save_root_folder + "seed" + str(seed) + "/"

        run_single_seed_test(dataset_generator=dataset_generator,
                             seed_save_folder=seed_save_folder,
                             seed=seed,
                             total_budget=total_budget,
                             ma_treshold=ma_threshold,
                             methods=methods)


def run_single_seed_test(dataset_generator,
                         seed_save_folder,
                         seed,
                         total_budget,
                         ma_treshold,
                         methods):
    try:
        os.makedirs(seed_save_folder)
    except OSError:
        if not os.path.isdir(seed_save_folder):
            raise

    m = dataset_generator.get_dataset_model(seed_folder=seed_save_folder,
                                            seed=seed,
                                            ma_treshold=ma_treshold)

    filename_rewards = seed_save_folder + "reward_histories.txt"

    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(filename_rewards, append_write)

    for method in methods:
        current_res = testWithFixedParameters(model=m,
                                          method=method.method_type,
                                          horizon=method.h,
                                          total_budget=total_budget,
                                          save_folder="{}{}/".format(seed_save_folder, method.method_folder_name),
                                          num_samples=method.num_samples,
                                          beta=method.beta)

        output_rewards.write(method.method_folder_name + '\n')
        output_rewards.write(str(current_res) + '\n')
    output_rewards.close()

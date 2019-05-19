import os

from src.TreePlanTester import testWithFixedParameters
from src.dataset_model.DatasetGenerator import DatasetGenerator
from src.enum.MethodEnum import Methods


def run(h,
        batch_size,
        total_budget,
        num_samples,
        beta,
        seeds,
        dataset_type,
        dataset_mode,
        dataset_root_folder,
        results_save_root_folder,
        ma_threshold,
        method):
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

        run_single_test(dataset_generator=dataset_generator,
                        seed_save_folder=seed_save_folder,
                        seed=seed,
                        total_budget=total_budget,
                        h=h,
                        beta=beta,
                        num_samples=num_samples,
                        ma_treshold=ma_threshold,
                        method=method)


def run_single_test(dataset_generator,
                    seed_save_folder,
                    seed,
                    total_budget,
                    h,
                    beta,
                    num_samples,
                    ma_treshold,
                    method):
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

    if method == Methods.Exact:
        method_string = 'exact'
    elif method == Methods.Anytime:
        method_string = 'anytime'
    else:
        raise Exception("Wrong method")

    current_res = testWithFixedParameters(model=m,
                                          method=method,
                                          horizon=h,
                                          total_budget=total_budget,
                                          save_folder="{}{}_h{}_beta{}/".format(seed_save_folder, method_string, h,
                                                                                beta),
                                          num_samples=num_samples,
                                          beta=beta)
    method_name = '{} h={} beta={}'.format(method_string, h, beta)
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(current_res) + '\n')
    output_rewards.close()

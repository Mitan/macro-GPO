import os

from src.dataset_model.DatasetGenerator import DatasetGenerator


def run(h_list,
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
    dataset_generator = DatasetGenerator(dataset_type=dataset_type, dataset_mode=dataset_mode,
                                         batch_size=batch_size)
    for h in h_list:
        for seed in seeds:
            pass




def TestScenario_beta(my_save_folder_root, seed, total_budget, h, beta_list,
                          num_samples, batch_size, time_slot, dataset_type, dataset_mode, ma_treshold):
        save_folder = my_save_folder_root + "seed" + str(seed) + "/"

        try:
            os.makedirs(save_folder)
        except OSError:
            if not os.path.isdir(save_folder):
                raise

        dataset_generator = DatasetGenerator(dataset_type=dataset_type, dataset_mode=dataset_mode,
                                             batch_size=batch_size)
        m = dataset_generator.get_dataset_model(root_folder=save_folder, seed=seed, ma_treshold=ma_treshold)

        filename_rewards = save_folder + "reward_histories.txt"
        if os.path.exists(filename_rewards):
            append_write = 'a'
        else:
            append_write = 'w'

        output_rewards = open(filename_rewards, append_write)
        for beta in beta_list:
            print(beta)
            current_res = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                                  total_budget=total_budget,
                                                  save_folder=save_folder + "beta" + str(beta) + "/",
                                                  num_samples=num_samples, beta=beta)
            method_name = 'beta=' + str(beta)
            output_rewards.write(method_name + '\n')
            output_rewards.write(str(current_res) + '\n')
        output_rewards.close()
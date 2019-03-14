import os

from src.enum.MethodEnum import Methods
from src.TreePlanTester import testWithFixedParameters
from src.dataset_model.DatasetGenerator import DatasetGenerator


def TestScenario_all_tests_road(my_save_folder_root, seed, total_budget, anytime_num_samples,
                                num_samples, batch_size, time_slot, dataset_type, dataset_mode, ma_treshold,
                                anytime_num_iterations):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    dataset_generator = DatasetGenerator(dataset_type=dataset_type, dataset_mode=dataset_mode,
                                         time_slot=time_slot, batch_size=batch_size)
    m = dataset_generator.get_dataset_model(root_folder=save_folder, seed=seed, ma_treshold=ma_treshold)

    filename_rewards = save_folder + "reward_histories.txt"
    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(filename_rewards, append_write)
    #

    lp = testWithFixedParameters(model=m, method=Methods.LP, horizon=1,
                                 total_budget=total_budget,
                                 save_folder=save_folder + "lp/",
                                 num_samples=num_samples)
    method_name = 'LP'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(lp) + '\n')

    qEI = testWithFixedParameters(model=m, method=Methods.qEI, horizon=1,
                                  total_budget=total_budget,
                                  save_folder=save_folder + "qEI/",
                                  num_samples=num_samples)
    method_name = 'myqEI'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(qEI) + '\n')

    PE = testWithFixedParameters(model=m, method=Methods.BucbPE, horizon=1,
                                 total_budget=total_budget,
                                 save_folder=save_folder + "pe/",
                                 num_samples=num_samples)
    method_name = 'PE'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(PE) + '\n')

    bucb = testWithFixedParameters(model=m, method=Methods.BUCB, horizon=1,
                                   total_budget=total_budget,
                                   save_folder=save_folder + "bucb/",
                                   num_samples=num_samples)

    method_name = 'BUCB'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(bucb) + '\n')
    
    mle_h = 4
    mle_4 = testWithFixedParameters(model=m, method=Methods.MLE, horizon=mle_h,
                                    total_budget=total_budget,
                                    save_folder=save_folder + "mle_h" + str(mle_h) + "/",
                                    num_samples=num_samples)
    method_name = 'MLE H=4'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(mle_4) + '\n')

    for h in range(1, 3 + 1):
        current_h = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                                            total_budget=total_budget,
                                            save_folder=save_folder + "h" + str(h) + "/",
                                            num_samples=anytime_num_samples,
                                            anytime_num_iterations=anytime_num_iterations)
        method_name = 'H=' + str(h)
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(current_h) + '\n')
    """
    h = 4
    h_4 = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=4,
                                  total_budget=total_budget,
                                  save_folder=save_folder + "h" + str(h) + "_b1_" + str(num_samples) + "/",
                                  num_samples=num_samples)
    method_name = 'H=4'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(h_4) + '\n')
    """
    output_rewards.close()


def TestScenario_branin(my_save_folder_root, seed, total_budget,
                           num_samples, batch_size, time_slot, dataset_type, dataset_mode, ma_treshold):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    dataset_generator = DatasetGenerator(dataset_type=dataset_type, dataset_mode=dataset_mode,
                                         time_slot=time_slot, batch_size=batch_size)
    m = dataset_generator.get_dataset_model(root_folder=save_folder, seed=seed, ma_treshold=ma_treshold)

    filename_rewards = save_folder + "reward_histories.txt"
    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(filename_rewards, append_write)

    ei = testWithFixedParameters(model=m, method=Methods.EI, horizon=1,
                                 total_budget=total_budget,
                                 save_folder=save_folder + "ei/",
                                 num_samples=num_samples)

    method_name = 'EI'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(ei) + '\n')

    h = 2
    h_2 = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                  total_budget=total_budget,
                                  save_folder=save_folder + "h" + str(h) + "_b1_" + str(num_samples) + "/",
                                  num_samples=num_samples)
    method_name = 'H=2'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(h_2) + '\n')

    h = 3
    h_3 = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                  total_budget=total_budget,
                                  save_folder=save_folder + "h" + str(h) + "_b1_" + str(num_samples) + "/",
                                  num_samples=num_samples)
    method_name = 'H=3'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(h_3) + '\n')

    h = 4
    h_4 = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                  total_budget=total_budget,
                                  save_folder=save_folder + "h" + str(h) + "_b1_" + str(num_samples) + "/",
                                  num_samples=num_samples)
    method_name = 'H=4'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(h_4) + '\n')
    output_rewards.close()


def TestScenario_all_tests(my_save_folder_root, seed, total_budget, anytime_num_samples,
                           num_samples, batch_size, time_slot, dataset_type, dataset_mode, ma_treshold):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    dataset_generator = DatasetGenerator(dataset_type=dataset_type, dataset_mode=dataset_mode,
                                         time_slot=time_slot, batch_size=batch_size)
    m = dataset_generator.get_dataset_model(root_folder=save_folder, seed=seed, ma_treshold=ma_treshold)

    filename_rewards = save_folder + "reward_histories.txt"
    if os.path.exists(filename_rewards):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(filename_rewards, append_write)
    #
    """
    anytime_h4 = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=4,
                                         total_budget=total_budget,
                                         save_folder=save_folder + "anytime_h4/",
                                         num_samples=anytime_num_samples)

    method_name = 'Anytime H=4'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(anytime_h4) + '\n')

    lp = testWithFixedParameters(model=m, method=Methods.LP, horizon=1,
                                 total_budget=total_budget,
                                 save_folder=save_folder + "lp_1/",
                                 num_samples=num_samples)
    method_name = 'LP'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(lp) + '\n')

    qEI = testWithFixedParameters(model=m, method=Methods.qEI, horizon=1,
                                  total_budget=total_budget,
                                  save_folder=save_folder + "qEI/",
                                  num_samples=num_samples)
    method_name = 'myqEI'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(qEI) + '\n')

    PE = testWithFixedParameters(model=m, method=Methods.BucbPE, horizon=1,
                                 total_budget=total_budget,
                                 save_folder=save_folder + "pe/",
                                 num_samples=num_samples)
    method_name = 'PE'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(PE) + '\n')

    bucb = testWithFixedParameters(model=m, method=Methods.BUCB, horizon=1,
                                   total_budget=total_budget,
                                   save_folder=save_folder + "bucb/",
                                   num_samples=num_samples)

    method_name = 'BUCB'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(bucb) + '\n')

    mle_4 = testWithFixedParameters(model=m, method=Methods.MLE, horizon=4,
                                    total_budget=total_budget,
                                    save_folder=save_folder + "mle_h4/",
                                    num_samples=num_samples)
    method_name = 'MLE H=4'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(mle_4) + '\n')

    for h in range(1, 4 + 1):
        current_h = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                            total_budget=total_budget,
                                            save_folder=save_folder + "h" + str(h) + "/",
                                            num_samples=num_samples)
        method_name = 'H=' + str(h)
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(current_h) + '\n')
    """

    ei = testWithFixedParameters(model=m, method=Methods.EI, horizon=1,
                                 total_budget=total_budget,
                                 save_folder=save_folder + "ei/",
                                 num_samples=num_samples)

    method_name = 'EI'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(ei) + '\n')

    h = 2
    h_2 = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                  total_budget=total_budget,
                                  save_folder=save_folder + "h" + str(h) + "_b1_" + str(num_samples) + "/",
                                  num_samples=num_samples)
    method_name = 'H=2'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(h_2) + '\n')

    h = 4
    h_4 = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                  total_budget=total_budget,
                                  save_folder=save_folder + "h" + str(h) + "_b1_" + str(num_samples) + "/",
                                  num_samples=num_samples)
    method_name = 'H=4'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(h_4) + '\n')
    """
    horizon = 4
    rollout = testWithFixedParameters(model=m, method=Methods.Rollout, horizon=horizon,
                                      total_budget=total_budget,
                                      save_folder=save_folder + "rollout_h" + str(horizon) + "_gamma1_ei_mod/",
                                      num_samples=num_samples)
    method_name = 'Rollout H=4'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(rollout) + '\n')
    """
    output_rewards.close()


def TestScenario_beta(my_save_folder_root, seed, total_budget, h, beta_list,
                      num_samples, batch_size, time_slot, dataset_type, dataset_mode, ma_treshold):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    dataset_generator = DatasetGenerator(dataset_type=dataset_type, dataset_mode=dataset_mode,
                                         time_slot=time_slot, batch_size=batch_size)
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

import os

from src.enum.MethodEnum import Methods
from TreePlanTester import testWithFixedParameters
from src.model.DatasetGenerator import DatasetGenerator


def TestScenario_all_tests(my_save_folder_root, seed, time_steps, anytime_num_samples,
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

    anytime_h4 = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=4,
                                         num_timesteps_test=time_steps,
                                         save_folder=save_folder + "anytime_h4/",
                                         num_samples=anytime_num_samples)
    method_name = 'Anytime H=4'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(anytime_h4) + '\n')

    lp = testWithFixedParameters(model=m, method=Methods.LP, horizon=1,
                                 num_timesteps_test=time_steps,
                                 save_folder=save_folder + "lp/",
                                 num_samples=num_samples)
    method_name = 'LP'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(lp) + '\n')

    qEI = testWithFixedParameters(model=m, method=Methods.qEI, horizon=1,
                                  num_timesteps_test=time_steps,
                                  save_folder=save_folder + "qEI/",
                                  num_samples=num_samples)
    method_name = 'myqEI'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(qEI) + '\n')

    PE = testWithFixedParameters(model=m, method=Methods.BucbPE, horizon=1,
                                 num_timesteps_test=time_steps,
                                 save_folder=save_folder + "pe/",
                                 num_samples=num_samples)
    method_name = 'PE'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(PE) + '\n')

    bucb = testWithFixedParameters(model=m, method=Methods.BUCB, horizon=1,
                                   num_timesteps_test=time_steps,
                                   save_folder=save_folder + "bucb/",
                                   num_samples=num_samples)

    method_name = 'BUCB'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(bucb) + '\n')

    mle_4 = testWithFixedParameters(model=m, method=Methods.MLE, horizon=4,
                                    num_timesteps_test=time_steps,
                                    save_folder=save_folder + "mle_h4/",
                                    num_samples=num_samples)
    method_name = 'MLE H=4'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(mle_4) + '\n')

    for h in range(1, 4):
        current_h = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                            num_timesteps_test=time_steps,
                                            save_folder=save_folder + "h" + str(h) + "/",
                                            num_samples=num_samples)
        method_name = 'H=' + str(h)
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(current_h) + '\n')

    """
    h_4 = testWithFixedParameters(model=m, method=Methods.Exact, horizon=4,
                                  num_timesteps_test=time_steps,
                                  save_folder=save_folder + "h" + str(4) + "/",
                                  num_samples=num_samples)
    method_name = 'H=4'
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(h_4) + '\n')
    """
    output_rewards.close()

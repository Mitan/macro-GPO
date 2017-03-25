import os

import numpy as np

from ResultsPlotter import PlotData
from TreePlanTester import testWithFixedParameters

# from MethodEnum import MethodEnum
from MethodEnum import Methods
from DatasetUtils import GenerateModelFromFile, GenerateSimulatedModel


def TestScenario_PE(my_save_folder_root, seed, time_steps, num_samples, batch_size, filename=None):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    assert filename is not None
    m = GenerateModelFromFile(filename)
    testWithFixedParameters(model=m, method=Methods.BUCB_PE, horizon=1,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "new_pe/",
                            num_samples=num_samples, batch_size=batch_size)


def TestScenario_H4(my_save_folder_root, h_max, seed, time_steps, num_samples, batch_size, filename=None):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    assert filename is not None
    m = GenerateModelFromFile(filename)
    h = 4
    testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "h" + str(h) + "/",
                            num_samples=num_samples, batch_size=batch_size)


def TestScenario_AnytimeMLE4(my_save_folder_root, seed, time_steps, batch_size, filename=None):
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    assert filename is not None
    m = GenerateModelFromFile(filename)
    h = 4

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "1_s250_50k_anytime_h" + str(h) + "/",
                            num_samples=2, batch_size=batch_size, anytime_iterations=2)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "2_s250_50k_anytime_h" + str(h) + "/",
                            num_samples=250, batch_size=batch_size, anytime_iterations=50000)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "3_s250_50k_anytime_h" + str(h) + "/",
                            num_samples=250, batch_size=batch_size, anytime_iterations=50000)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "4_s250_50k_anytime_h" + str(h) + "/",
                            num_samples=250, batch_size=batch_size, anytime_iterations=50000)

    #####################
    """
    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s150_750_anytime_h" + str(h) + "/",
                            num_samples=150, batch_size=batch_size, anytime_iterations=750)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s200_750_anytime_h" + str(h) + "/",
                            num_samples=200, batch_size=batch_size, anytime_iterations=750)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s250_750_anytime_h" + str(h) + "/",
                            num_samples=250, batch_size=batch_size, anytime_iterations=750)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s300_750_anytime_h" + str(h) + "/",
                            num_samples=300, batch_size=batch_size, anytime_iterations=750)
    # 1000

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s150_1000_anytime_h" + str(h) + "/",
                            num_samples=150, batch_size=batch_size, anytime_iterations=1000)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s200_1000_anytime_h" + str(h) + "/",
                            num_samples=200, batch_size=batch_size, anytime_iterations=1000)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s250_1000_anytime_h" + str(h) + "/",
                            num_samples=250, batch_size=batch_size, anytime_iterations=1000)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s300_1000_anytime_h" + str(h) + "/",
                            num_samples=300, batch_size=batch_size, anytime_iterations=1000)

    # 1200
    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s150_1200_anytime_h" + str(h) + "/",
                            num_samples=150, batch_size=batch_size, anytime_iterations=1200)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s200_1200_anytime_h" + str(h) + "/",
                            num_samples=200, batch_size=batch_size, anytime_iterations=1200)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s250_1200_anytime_h" + str(h) + "/",
                            num_samples=250, batch_size=batch_size, anytime_iterations=1200)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s300_1200_anytime_h" + str(h) + "/",
                            num_samples=300, batch_size=batch_size, anytime_iterations=1200)

    # 1500
    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s150_1500_anytime_h" + str(h) + "/",
                            num_samples=150, batch_size=batch_size, anytime_iterations=1500)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s200_1500_anytime_h" + str(h) + "/",
                            num_samples=200, batch_size=batch_size, anytime_iterations=1500)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s250_1500_anytime_h" + str(h) + "/",
                            num_samples=250, batch_size=batch_size, anytime_iterations=1500)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s300_1500_anytime_h" + str(h) + "/",
                            num_samples=300, batch_size=batch_size, anytime_iterations=1500)

    # 2000

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s150_2000_anytime_h" + str(h) + "/",
                            num_samples=150, batch_size=batch_size, anytime_iterations=2000)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s200_2000_anytime_h" + str(h) + "/",
                            num_samples=200, batch_size=batch_size, anytime_iterations=2000)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s250_2000_anytime_h" + str(h) + "/",
                            num_samples=250, batch_size=batch_size, anytime_iterations=2000)

    testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "s300_2000_anytime_h" + str(h) + "/",
                            num_samples=300, batch_size=batch_size, anytime_iterations=2000)
    """

    """
    testWithFixedParameters(model=m, method=Methods.MLE, horizon=h,
                            num_timesteps_test=time_steps,
                            save_folder=save_folder + "mle_h" + str(h) + "/",
                            num_samples=num_samples, batch_size=batch_size)
    """


def TestScenario(my_save_folder_root, h_max, seed, time_steps, num_samples, batch_size, filename=None):
    result_graphs = []

    # eps = 10 ** 10
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    # this model is for observed values
    length_scale = (0.25, 0.25)
    signal_variance = 1.0
    noise_variance = 0.00001
    predict_range = ((-0.25, 2.25), (-0.25, 2.25))
    num_samples_grid = (50, 50)

    # file for storing reward histories
    # so that later we can plot only some of them

    output_rewards = open(save_folder + "reward_histories.txt", 'w')

    if filename is not None:
        m = GenerateModelFromFile(filename)
    else:
        m = GenerateSimulatedModel(length_scale=np.array(length_scale), signal_variance=signal_variance,
                                   seed=seed, noise_variance=noise_variance, save_folder=save_folder,
                                   predict_range=predict_range, num_samples=num_samples_grid)

    # can't apply qEI to single-point
    if batch_size > 1:
        method_name = 'qEI'
        qEI = testWithFixedParameters(model=m, method=Methods.qEI, horizon=1, num_timesteps_test=time_steps,
                                      save_folder=save_folder + "qEI/",
                                      num_samples=num_samples, batch_size=batch_size)
        result_graphs.append([method_name, qEI])
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(qEI) + '\n')

    method_name = 'Myopic DB-GP-UCB'
    myopic_ucb = testWithFixedParameters(model=m, method=Methods.MyopicUCB, horizon=1, num_timesteps_test=time_steps,
                                         save_folder=save_folder + "h1/",
                                         num_samples=num_samples, batch_size=batch_size)
    result_graphs.append([method_name, myopic_ucb])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(myopic_ucb) + '\n')

    # for h in range(2, h_max+1):
    for h in range(h_max, 1, -1):
        # print h
        method_name = 'H = ' + str(h)
        current_h_result = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                                   num_timesteps_test=time_steps,
                                                   save_folder=save_folder + "h" + str(h) + "/",
                                                   num_samples=num_samples, batch_size=batch_size)
        result_graphs.append([method_name, current_h_result])
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(current_h_result) + '\n')

    method_name = 'MLE H = 3'
    mle = testWithFixedParameters(model=m, method=Methods.MLE, horizon=3, num_timesteps_test=time_steps,
                                  save_folder=save_folder + "mle_h3/",
                                  num_samples=num_samples, batch_size=batch_size)
    result_graphs.append([method_name, mle])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(mle) + '\n')

    method_name = 'Anytime H = 3'
    anytime = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=3, num_timesteps_test=time_steps,
                                      save_folder=save_folder + "anytime_h3/",
                                      num_samples=num_samples, batch_size=batch_size)
    result_graphs.append([method_name, anytime])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(anytime) + '\n')

    output_rewards.close()
    PlotData(result_graphs, save_folder)


def TestScenario_Beta(my_save_folder_root, seed, time_steps, num_samples, batch_size, beta_list, test_horizon,
                      filename=None):
    result_graphs = []

    # test_horizon = 3

    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    output_rewards = open(save_folder + "reward_histories.txt", 'w')

    assert filename is not None

    m = GenerateModelFromFile(filename)

    for beta in beta_list:
        method_name = 'beta = ' + str(beta)
        current_h_result = testWithFixedParameters(model=m, method=Methods.Exact, horizon=test_horizon,
                                                   num_timesteps_test=time_steps,
                                                   save_folder=save_folder + "beta" + str(beta) + "/",
                                                   num_samples=num_samples, batch_size=batch_size, beta=beta)
        result_graphs.append([method_name, current_h_result])
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(current_h_result) + '\n')

    output_rewards.close()
    PlotData(result_graphs, save_folder)

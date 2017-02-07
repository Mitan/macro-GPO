import os

import numpy as np

from ResultsPlotter import PlotData
from TreePlanTester import testWithFixedParameters

# from MethodEnum import MethodEnum
from MethodEnum import Methods
from DatasetUtils import GenerateModelFromFile, GenerateSimulatedModel, GenerateRoadModelFromFile
from HypersStorer import SimulatedHyperStorer


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


def TestScenario(my_save_folder_root, h_max, seed, time_steps, num_samples, batch_size, filename=None):
    result_graphs = []

    # eps = 10 ** 10
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    # file for storing reward histories
    # so that later we can plot only some of them
    output_rewards = open(save_folder + "reward_histories.txt", 'w')

    # Generation of simulated model
    """
    if filename is not None:
        m = GenerateModelFromFile(filename)

    else:
        hyper_storer = SimulatedHyperStorer()
        m = GenerateSimulatedModel(length_scale=np.array(hyper_storer.length_scale),
                                   signal_variance=hyper_storer.signal_variance,
                                   seed=seed, noise_variance=hyper_storer.noise_variance, save_folder=save_folder,
                                   predict_range=hyper_storer.grid_domain, num_samples=hyper_storer.num_samples_grid,
                                   mean_function=hyper_storer.mean_function)
    """

    filename = '../datasets/slot44/tlog44.dom'
    m = GenerateRoadModelFromFile(filename)

    # todo note
    m.SelectMacroActions(batch_size)

    start_location = m.GetRandomStartLocation(batch_size=batch_size)

    with  open(save_folder + "start_location.txt", 'w') as f:
        f.write(str(start_location[0]) + " " + str(start_location[1]))

    # can't apply qEI to single-point

    """
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
    anytime = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=1, num_timesteps_test=time_steps,
                                      save_folder=save_folder + "anytime_h3/",
                                      num_samples=num_samples, batch_size=batch_size, start_location=start_location)
    result_graphs.append([method_name, anytime])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(anytime) + '\n')
    """

    if batch_size > 1:
        method_name = 'qEI'
        qEI = testWithFixedParameters(model=m, method=Methods.qEI, horizon=1, num_timesteps_test=time_steps,
                                      save_folder=save_folder + "qEI/",
                                      num_samples=num_samples, batch_size=batch_size, start_location=start_location)
        result_graphs.append([method_name, qEI])
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(qEI) + '\n')

    method_name = 'Myopic DB-GP-UCB'
    myopic_ucb = testWithFixedParameters(model=m, method=Methods.MyopicUCB, horizon=1, num_timesteps_test=time_steps,
                                         save_folder=save_folder + "h1/",
                                         num_samples=num_samples, batch_size=batch_size, start_location=start_location)
    result_graphs.append([method_name, myopic_ucb])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(myopic_ucb) + '\n')

    method_name = 'MLE H = 3'
    mle = testWithFixedParameters(model=m, method=Methods.MLE, horizon=3, num_timesteps_test=time_steps,
                                  save_folder=save_folder + "mle_h3/",
                                  num_samples=num_samples, batch_size=batch_size, start_location=start_location)
    result_graphs.append([method_name, mle])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(mle) + '\n')

    # for h in range(2, h_max+1):
    for h in range(2, h_max + 1):
        # print h
        method_name = 'Anytime H = ' + str(h)
        current_h_result = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=h,
                                                   num_timesteps_test=time_steps,
                                                   save_folder=save_folder + "anytime_h" + str(h) + "/",
                                                   num_samples=num_samples, batch_size=batch_size,
                                                   start_location=start_location)
        result_graphs.append([method_name, current_h_result])
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(current_h_result) + '\n')


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

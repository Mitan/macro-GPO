import os

from ResultsPlotter import PlotData
from TreePlanTester import testWithFixedParameters
from GaussianProcess import SquareExponential

from GaussianProcess import GaussianProcess


def TestScenario(my_save_folder_root, h_max, seed, time_steps):

    result_graphs = []

    #eps = 10 ** 10
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    length_scale = (0.1, 0.1)
    covariance_function = SquareExponential(length_scale, 1)
    gpgen = GaussianProcess(covariance_function)
    m = gpgen.GPGenerate(predict_range=((0, 1), (0, 1)), num_samples=(20, 20), seed=seed)
    # write the dataset to file
    m.WriteToFile(save_folder + "dataset.txt")

    for h in range(1, h_max):
        # print h
        current_h_result = testWithFixedParameters(model=m, horizon=h, num_timesteps_test=time_steps,
                                                   length_scale=length_scale,
                                                   save_folder=save_folder + "h" + str(h) + "/",
                                                   preset=False, MCTS=False, Randomized= True)
        result_graphs.append(['H = ' + str(h), current_h_result])

    """
    ucb = TestWithFixedParameters(initial_state=my_initial_state, horizon=1, batch_size=b, alg_type='UCB',
                                  beta=beta, simulated_function=simulated_func,
                                          save_folder=my_save_folder_root + 'ucb' + "/", num_timesteps_test= time_steps)
    result_graphs.append(['UCB', ucb])

    qei = TestWithFixedParameters(initial_state=my_initial_state, horizon=1, batch_size=b, alg_type='qEI',
                                  beta=beta, simulated_function=simulated_func,
                                          save_folder=my_save_folder_root + 'ei' + "/", num_timesteps_test= time_steps)
    result_graphs.append(['qEI', qei])


    # non-myopic part
    # h = 2
    my_save_folder = my_save_folder_root + "h" + str(2)
    non_myopic_2 = TestWithFixedParameters(initial_state=my_initial_state, horizon=2, batch_size=b,
                                                   alg_type='Non-myopic',
                                           beta=beta,
                                                   simulated_function=simulated_func,
                                                   save_folder=my_save_folder + '_non-myopic' + "/", num_timesteps_test= time_steps)
    result_graphs.append(['H=2', non_myopic_2])


    #MLE h = 3
    my_save_folder = my_save_folder_root + "h" + str(3)
    mle = TestWithFixedParameters(initial_state=my_initial_state, horizon=3, batch_size=b, alg_type='MLE',
                                  beta=beta, simulated_function=simulated_func,
                                          save_folder=my_save_folder + '_mle' + "/", num_timesteps_test= time_steps)
    result_graphs.append(['ML H = 3', mle])


    # h = 3
    my_save_folder = my_save_folder_root + "h" + str(3)
    non_myopic_3 = TestWithFixedParameters(initial_state=my_initial_state, horizon=3, batch_size=b,
                                                   alg_type='Non-myopic',
                                           beta=beta,
                                                   simulated_function=simulated_func,
                                                   save_folder=my_save_folder + '_non-myopic' + "/", num_timesteps_test= time_steps)
    result_graphs.append(['H=3', non_myopic_3])


    print datetime.now()
    # h = 4
    my_save_folder = my_save_folder_root + "h" + str(4)
    non_myopic_4 = TestWithFixedParameters(initial_state=my_initial_state, horizon=4, batch_size=b,
                                                   alg_type='Non-myopic',
                                                   my_samples_count_func=f, beta=beta,
                                                   simulated_function=simulated_func,
                                                   save_folder=my_save_folder + '_non-myopic' + "/", num_timesteps_test= time_steps)
    result_graphs.append(['H=4', non_myopic_4])
    print datetime.now()
    """

    PlotData(result_graphs, save_folder)



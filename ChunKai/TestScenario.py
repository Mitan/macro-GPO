from ResultsPlotter import PlotData
from TreePlanTester import TestWithFixedParameters


def TestScenario(b, beta, location, simulated_func, my_save_folder_root):
    """
    :param b: batch size
    :param beta: beta from ucb reward function
    :param location:  initial location for agents
    :param simulated_func: function info
    :param save_trunk: root folder
    :return:
    """
    result_graphs = []
    my_initial_state = location
    time_steps = 20

    # folder where we can put results of methods

    # these algorithms are myopic
    #  nodes function
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
    """

    # h = 3
    my_save_folder = my_save_folder_root + "h" + str(3)
    non_myopic_3 = TestWithFixedParameters(initial_state=my_initial_state, horizon=3, batch_size=b,
                                                   alg_type='Non-myopic',
                                           beta=beta,
                                                   simulated_function=simulated_func,
                                                   save_folder=my_save_folder + '_non-myopic' + "/", num_timesteps_test= time_steps)
    result_graphs.append(['H=3', non_myopic_3])

    """
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
    PlotData(result_graphs, my_save_folder_root)
    """
    return non_myopic_3
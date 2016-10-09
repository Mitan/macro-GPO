import os

from ResultsPlotter import PlotData
from TreePlanTester import testWithFixedParameters
from GaussianProcess import SquareExponential

from GaussianProcess import GaussianProcess
# from MethodEnum import MethodEnum
from MethodEnum import Methods


def GenerateSimulatedModel(length_scale, signal_variance, save_folder, seed):
    covariance_function = SquareExponential(length_scale, signal_variance=signal_variance)
    gpgen = GaussianProcess(covariance_function, noise_variance=0)
    m = gpgen.GPGenerate(predict_range=((0, 1), (0, 1)), num_samples=(20, 20), seed=seed)
    # write the dataset to file
    m.WriteToFile(save_folder + "dataset.txt")
    return m


def TestScenario(my_save_folder_root, h_max, seed, time_steps, num_samples, batch_size):
    result_graphs = []

    # eps = 10 ** 10
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    # this model is for observed values
    length_scale = (0.1, 0.1)
    signal_variance = 1.0
    noise_variance = 0.05

    # this model contains noiseless values
    m = GenerateSimulatedModel(length_scale=length_scale, signal_variance=signal_variance,
                               seed=seed)

    # todo fix horizon to 1
    myopic_ucb = testWithFixedParameters(model=m, method=Methods.MyopicUCB, horizon=2, num_timesteps_test=time_steps,
                                         save_folder=save_folder + "h1/",
                                         num_samples=num_samples, batch_size=batch_size)
    result_graphs.append(['Myopic DB-GP-UCB', myopic_ucb])
    """
    for h in range(2, h_max):
        # print h
        current_h_result = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                                   num_timesteps_test=time_steps,
                                                   length_scale=length_scale,
                                                   save_folder=save_folder + "h" + str(h) + "/",
                                                   preset=False, num_samples=num_samples, batch_size=batch_size)
        result_graphs.append(['H = ' + str(h), current_h_result])

    anytime = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=3, num_timesteps_test=time_steps,
                                      length_scale=length_scale,
                                      save_folder=save_folder + "anytime_h3/",
                                      preset=False, num_samples=num_samples, batch_size=batch_size)
    result_graphs.append(['Anytime H = 3', anytime])

    mle = testWithFixedParameters(model=m, method=Methods.MLE, horizon=3, num_timesteps_test=time_steps,
                                  length_scale=length_scale,
                                  save_folder=save_folder + "mle_h3/",
                                  preset=False, num_samples=num_samples, batch_size=batch_size)
    result_graphs.append(['MLE H = 3', mle])
    """
    PlotData(result_graphs, save_folder)

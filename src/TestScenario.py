import os
import numpy as np

from ResultsPlotter import PlotData
from TreePlanTester import testWithFixedParameters
from GaussianProcess import SquareExponential

from GaussianProcess import GaussianProcess
# from MethodEnum import MethodEnum
from MethodEnum import Methods


def GenerateSimulatedModel(length_scale, signal_variance, noise_variance, save_folder, seed):
    covariance_function = SquareExponential(length_scale, signal_variance=signal_variance, noise_variance=noise_variance)
    # Generate a drawn vector from GP with noise
    gpgen = GaussianProcess(covariance_function)
    m = gpgen.GPGenerate(predict_range=((0, 1), (0, 1)), num_samples=(20, 20), seed=seed, noiseVariance=noise_variance)
    # write the dataset to file
    m.WriteToFile(save_folder + "dataset.txt")
    return m


def GenerateModelFromFile(filename):
    m = GaussianProcess.GPGenerateFromFile(filename)

    return m


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
    length_scale = (0.1, 0.1)
    signal_variance = 1.0
    noise_variance = 0.05

    # this model contains noiseless values
    if filename is not None:
        m = GenerateModelFromFile(filename)
    else:
        m = GenerateSimulatedModel(length_scale=np.array(length_scale), signal_variance=signal_variance,
                                   seed=seed, noise_variance=noise_variance, save_folder=save_folder)

    myopic_ucb = testWithFixedParameters(model=m, method=Methods.MyopicUCB, horizon=1, num_timesteps_test=time_steps,
                                         save_folder=save_folder + "h1/",
                                         num_samples=num_samples, batch_size=batch_size)
    result_graphs.append(['Myopic DB-GP-UCB', myopic_ucb])

    for h in range(2, h_max):
        # print h
        current_h_result = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                                   num_timesteps_test=time_steps,
                                                   save_folder=save_folder + "h" + str(h) + "/",
                                                   num_samples=num_samples, batch_size=batch_size)
        result_graphs.append(['H = ' + str(h), current_h_result])

    mle = testWithFixedParameters(model=m, method=Methods.MLE, horizon=3, num_timesteps_test=time_steps,
                                  save_folder=save_folder + "mle_h3/",
                                  num_samples=num_samples, batch_size=batch_size)
    result_graphs.append(['MLE H = 3', mle])

    anytime = testWithFixedParameters(model=m, method=Methods.Anytime, horizon=3, num_timesteps_test=time_steps,
                                      length_scale=length_scale,
                                      save_folder=save_folder + "anytime_h3/",
                                      preset=False, num_samples=num_samples, batch_size=batch_size)
    result_graphs.append(['Anytime H = 3', anytime])

    # can't apply qEI to single-point
    if batch_size > 1:
        qEI = testWithFixedParameters(model=m, method=Methods.qEI, horizon=1, num_timesteps_test=time_steps,
                                      save_folder=save_folder + "qEI/",
                                      num_samples=num_samples, batch_size=batch_size)
        result_graphs.append(['qEI', qEI])

    PlotData(result_graphs, save_folder)

import os

from ResultsPlotter import PlotData
from TreePlanTester import testWithFixedParameters
from GaussianProcess import SquareExponential

from GaussianProcess import GaussianProcess
from MethodEnum import MethodEnum


def TestScenario(my_save_folder_root, h_max, seed, time_steps, num_samples):
    result_graphs = []

    # eps = 10 ** 10
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

    myopic_ucb = testWithFixedParameters(model=m, method=MethodEnum.MyopicUCB, horizon=1, num_timesteps_test=time_steps,
                                         length_scale=length_scale,
                                         save_folder=save_folder + "h1/",
                                         preset=False, num_samples=num_samples)
    result_graphs.append(['Myopic DB-GP-UCB', myopic_ucb])

    for h in range(2, h_max):
        # print h
        current_h_result = testWithFixedParameters(model=m, method=MethodEnum.Exact, horizon=h,
                                                   num_timesteps_test=time_steps,
                                                   length_scale=length_scale,
                                                   save_folder=save_folder + "h" + str(h) + "/",
                                                   preset=False, num_samples=num_samples)
        result_graphs.append(['H = ' + str(h), current_h_result])

    anytime = testWithFixedParameters(model=m, method=MethodEnum.Anytime, horizon=3, num_timesteps_test=time_steps,
                                      length_scale=length_scale,
                                      save_folder=save_folder + "anytime_h3/",
                                      preset=False, num_samples=num_samples)
    result_graphs.append(['Anytime H = 3', anytime])

    mle = testWithFixedParameters(model=m, method=MethodEnum.MLE, horizon=3, num_timesteps_test=time_steps,
                                  length_scale=length_scale,
                                  save_folder=save_folder + "mle_h3/",
                                  preset=False, num_samples=num_samples)
    result_graphs.append(['MLE H = 3', mle])

    PlotData(result_graphs, save_folder)

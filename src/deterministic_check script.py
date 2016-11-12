import os
import sys

from GaussianProcess import GaussianProcess
from ResultsPlotter import PlotData
from TreePlanTester import testWithFixedParameters

# from MethodEnum import MethodEnum
from MethodEnum import Methods

if __name__ == '__main__':

    my_save_folder_root = "../tests/full_dynamic_b2_dcheck/"
    # max horizon
    h_max = 5
    # time steps
    t = 10

    batch_size = 2

    # number of samples per stage
    num_samples = 50

    args = sys.argv

    filename = "./debug_dataset.txt"

    seed = int(args[1])

    result_graphs = []
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    # file for storing reward histories
    # so that later we can plot only some of them
    output_rewards = open(save_folder + "reward_histories.txt", 'w')

    m = GaussianProcess.GPGenerateFromFile(filename)

    method_name = 'Myopic DB-GP-UCB'
    myopic_ucb = testWithFixedParameters(model=m, method=Methods.MyopicUCB, horizon=1,
                                         num_timesteps_test=t,
                                         save_folder=save_folder + "h1/",
                                         num_samples=num_samples, batch_size=batch_size)
    result_graphs.append([method_name, myopic_ucb])
    output_rewards.write(method_name + '\n')
    output_rewards.write(str(myopic_ucb) + '\n')

    for h in range(2, h_max):
        # print h
        method_name = 'H = ' + str(h)
        current_h_result = testWithFixedParameters(model=m, method=Methods.Exact, horizon=h,
                                                   num_timesteps_test=t,
                                                   save_folder=save_folder + "h" + str(h) + "/",
                                                   num_samples=num_samples, batch_size=batch_size)
        result_graphs.append([method_name, current_h_result])
        output_rewards.write(method_name + '\n')
        output_rewards.write(str(current_h_result) + '\n')

    # can't apply qEI to single-point
    output_rewards.close()
    PlotData(result_graphs, save_folder)

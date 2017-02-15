from ResultsPlotter import PlotData
import numpy as np
from StringIO import StringIO


def CalculateTotalRewards(batch_size, root_path, methods, method_names, seeds):
    steps = 20 / batch_size

    results = []

    # coeff_file = open(root_path + 'g_coefficients.txt', 'w')

    for index, method in enumerate(methods):
        number_of_location = 0
        sum_coefficients = 0
        results_for_method = np.zeros((steps,))
        for seed in seeds:
            # counter

            seed_folder = root_path + 'seed' + str(seed) + '/'
            file_path = seed_folder + method + '/summary.txt'
            # print file_path
            try:
                # normalized
                a = (open(file_path).readlines()[-1])[27: -2]
                # print a
                # print file_path
                number_of_location += 1
            except:
                print file_path
                continue

            a = StringIO(a)

            rewards = np.genfromtxt(a, delimiter=",")
            results_for_method = np.add(results_for_method, rewards)

            # todo need to adjust coefficient
            # g_coefficient = GetGCoefficient(seed_folder, method)
            g_coefficient = 0
            sum_coefficients += g_coefficient

        # check that we collected data for every location
        # print method
        # print number_of_location, len(seeds)
        assert number_of_location == len(seeds)
        # print results_for_method
        results_for_method = results_for_method / number_of_location
        result = [method_names[index], results_for_method.tolist()]
        results.append(result)

        mean_coefficient = sum_coefficients / number_of_location
        # coeff_file.write(method_names[index] + ' ' + str(mean_coefficient) + '\n')

    PlotData(results, root_path)
    # coeff_file.close()


def GetSimulatedTotalRewards():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../releaseTests/simulated/rewards-sAD/'
    methods = ['h1', 'h2', 'h3', 'h4', 'anytime_h3', 'mle_h3', 'qEI']
    method_names = ['H = 1', 'H = 2', 'H = 3', 'H = 4', 'Anytime', 'MLE H = 3', 'qEI']
    CalculateTotalRewards(batch_size, root_path, methods, method_names, seeds)


def GetRoadTotalRewards():
    seeds = range(36)
    seeds = list(set(seeds) - set([31]))
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'mle_h3', 'qEI']
    method_names = ['Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'MLE H = 3', 'qEI']

    root_path = '../releaseTests/road/b5-18-log/'
    CalculateTotalRewards(batch_size, root_path, methods, method_names, seeds)


if __name__ == "__main__":
    # GetRoadTotalRewards()
    GetSimulatedTotalRewards()

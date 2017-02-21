from src.ResultsPlotter import PlotData
import numpy as np

from src.DatasetUtils import GetAllMeasurements, GetAccumulatedRewards, GenerateModelFromFile, GenerateRoadModelFromFile


def RoadRewards(batch_size, root_path, methods, method_names, seeds):
    """
    seeds = range(36)
    seeds = list(set(seeds) - set([31]))
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'mle_h3', 'qEI']
    method_names = ['Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'MLE H = 3', 'qEI']

    root_path = '../../releaseTests/road/b5-18-log/'
    """
    steps = 20 / batch_size

    len_seeds = len(seeds)
    results = []

    dataset_file_name = '../../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(dataset_file_name)
    model_mean = m.mean

    scaled_model_mean = np.array([(1 + batch_size * i) * model_mean for i in range(steps + 1)])
    print scaled_model_mean

    for index, method in enumerate(methods):
        number_of_location = 0

        # +1 initial point
        results_for_method = np.zeros((steps + 1,))
        for seed in seeds:

            seed_folder = root_path + 'seed' + str(seed) + '/'
            try:
                # all measurements, unnormalized
                measurements = GetAllMeasurements(seed_folder, method, batch_size)
                number_of_location += 1
            except:
                print seed_folder, method
                continue

            rewards = GetAccumulatedRewards(measurements, batch_size)
            results_for_method = np.add(results_for_method, rewards)

        # check that we collected data for every location
        # print method
        # print number_of_location, len(seeds)
        assert number_of_location == len(seeds)
        # print results_for_method
        results_for_method = results_for_method / len_seeds
        scaled_results = results_for_method - scaled_model_mean
        result = [method_names[index], scaled_results.tolist()]
        results.append(result)
    PlotData(results, root_path)


def SimulatedRewards(batch_size, root_path, methods, method_names, seeds):
    """
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/rewards-sAD/'
    methods = ['h1', 'h2', 'h3', 'h4', 'anytime_h3', 'mle_h3', 'qEI']
    method_names = ['H = 1', 'H = 2', 'H = 3', 'H = 4', 'Anytime', 'MLE H = 3', 'qEI']
    """
    steps = 20 / batch_size

    len_seeds = len(seeds)
    results = []

    sum_model_mean = 0
    for seed in seeds:
        seed_dataset_path = root_path + 'seed' + str(seed) + '/dataset.txt'
        m = GenerateModelFromFile(seed_dataset_path)
        sum_model_mean += m.mean

    average_model_mean = sum_model_mean / len_seeds
    scaled_model_mean = np.array([(1 + batch_size * i) * average_model_mean for i in range(steps + 1)])
    print scaled_model_mean

    for index, method in enumerate(methods):
        number_of_location = 0

        # +1 initial point
        results_for_method = np.zeros((steps + 1,))
        for seed in seeds:

            seed_folder = root_path + 'seed' + str(seed) + '/'
            try:
                # all measurements, unnormalized
                measurements = GetAllMeasurements(seed_folder, method, batch_size)
                number_of_location += 1
            except:
                print seed_folder, method
                continue

            rewards = GetAccumulatedRewards(measurements, batch_size)
            results_for_method = np.add(results_for_method, rewards)

        # check that we collected data for every location
        # print method
        # print number_of_location, len(seeds)
        assert number_of_location == len(seeds)
        # print results_for_method
        results_for_method = results_for_method / len_seeds
        scaled_results = results_for_method - scaled_model_mean
        result = [method_names[index], scaled_results.tolist()]
        results.append(result)
    PlotData(results, root_path)


"""
# todo unused
def CalculateTotalRewards(batch_size, root_path, methods, method_names, seeds):
    steps = 20 / batch_size

    results = []

    # coeff_file = open(root_path + 'g_coefficients.txt', 'w')

    for index, method in enumerate(methods):
        number_of_location = 0

        results_for_method = np.zeros((steps,))
        for seed in seeds:
            # counter

            seed_folder = root_path + 'seed' + str(seed) + '/'
            file_path = seed_folder + method + '/summary.txt'
            # print file_path
            try:
                # normalized
                a = (open(file_path).readlines()[-1])
                # print a
                a = a[27: -2]
                # print a
                # print file_path
                number_of_location += 1
            except:
                print file_path
                continue

            a = StringIO(a)

            rewards = np.genfromtxt(a, delimiter=",")
            results_for_method = np.add(results_for_method, rewards)


        # check that we collected data for every location
        # print method
        # print number_of_location, len(seeds)
        assert number_of_location == len(seeds)
        # print results_for_method
        results_for_method = results_for_method / number_of_location
        result = [method_names[index], results_for_method.tolist()]
        results.append(result)

    PlotData(results, root_path)
"""


##### Simulated ####
def GetSimulatedTotalRewards():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/rewards-sAD/'
    methods = ['h1', 'h2', 'h3', 'h4', 'anytime_h3', 'mle_h3', 'qEI']
    method_names = ['H = 1', 'H = 2', 'H = 3', 'H = 4', 'Anytime', 'MLE H = 3', 'qEI']
    SimulatedRewards(batch_size, root_path, methods, method_names, seeds)


def GetSimulatedBeta2Rewards():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/simulatedBeta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    SimulatedRewards(batch_size, root_path, methods, method_names, seeds)


def GetSimulatedBeta3Rewards():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/simulatedBeta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    SimulatedRewards(batch_size, root_path, methods, method_names, seeds)


####### Road ########
def GetRoadBeta2Rewards():
    seeds = range(0, 35)
    seeds = list(set(seeds) - set([30]))
    root_path = '../../releaseTests/road/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    RoadRewards(batch_size, root_path, methods, method_names, seeds)


def GetRoadBeta3Rewards():
    seeds = range(0, 32)
    # seeds = list(set(seeds) - set([27,31]))
    # seeds = list(set(seeds) - set([0,14]))
    seeds = list(set(seeds) - set([5]))
    root_path = '../../releaseTests/road/beta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    RoadRewards(batch_size, root_path, methods, method_names, seeds)


def GetRoadTotalRewards():
    seeds = range(36)
    seeds = list(set(seeds) - set([31]))
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4', 'mle_h3', 'qEI', 'pe']
    method_names = ['Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'Anytime H = 4', 'MLE H = 3', 'qEI', 'BUCB-PE']

    root_path = '../../releaseTests/road/b5-18-log/'
    RoadRewards(batch_size, root_path, methods, method_names, seeds)


if __name__ == "__main__":
    """
    GetSimulatedTotalRewards()
    GetSimulatedBeta2Rewards()
    GetSimulatedBeta3Rewards()

    GetRoadBeta2Rewards()
    GetRoadBeta3Rewards()
    GetRoadTotalRewards()
    """
    # GetSimulatedBeta2Rewards()
    # GetSimulatedBeta3Rewards()
    # GetRoadBeta3Rewards()
    # GetRoadBeta2Rewards()
    GetRoadTotalRewards()

from StringIO import StringIO
import numpy as np

from src.DatasetUtils import GenerateRoadModelFromFile, GetAllMeasurements, GetMaxValues, GenerateModelFromFile
from src.ResultsPlotter import PlotData


def RoadRegrets(batch_size, root_path, methods, method_names, seeds, output_filename):
    """
    root_path = '../../releaseTests/road/b5-18-log/'
    seeds = range(35)
    seeds = list(set(seeds) - set([31]))
    batch_size = 5
    methods = ['qEI', 'h1', 'anytime_h2', 'anytime_h3', 'mle_h3']
    method_names = ['qEI', 'Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'MLE H = 3']
    """
    len_seeds = len(seeds)
    steps = 20 / batch_size

    dataset_file_name = '../../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(dataset_file_name)
    model_max = m.GetMax()

    results = []

    # for every method
    for index, method in enumerate(methods):

        # +1 for initial stage
        results_for_method = np.zeros((steps + 1,))

        for seed in seeds:
            seed_folder = root_path + 'seed' + str(seed) + '/'
            measurements = GetAllMeasurements(seed_folder, method, batch_size)
            max_found_values = GetMaxValues(measurements, batch_size)
            assert max_found_values.shape == results_for_method.shape

            results_for_method = np.add(results_for_method, max_found_values)

        results_for_method = results_for_method / len_seeds
        regrets = [model_max - res for res in results_for_method.tolist()]
        result = [method_names[index], regrets]
        results.append(result)
        print result

    PlotData(results=results, output_file_name=output_filename, isTotalReward=False, isRoad=True)


def SimulatedRegrets(batch_size, root_path, methods, method_names, seeds, output_filename):
    """
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/rewards-sAD/'
    methods = ['h1', 'h2', 'h3', 'h4', 'anytime_h3', 'mle_h3', 'qEI']
    method_names = ['H = 1', 'H = 2', 'H = 3', 'H = 4', 'Anytime', 'MLE H = 3', 'qEI']
    """
    len_seeds = len(seeds)
    steps = 20 / batch_size

    results = []

    # average regret is average max - average reward
    # average max is sum over all max seeds / len
    sum_model_max = 0
    for seed in seeds:
        seed_dataset_path = root_path + 'seed' + str(seed) + '/dataset.txt'
        m = GenerateModelFromFile(seed_dataset_path)
        sum_model_max += m.GetMax()

    average_model_max = sum_model_max / len_seeds

    # for every method
    for index, method in enumerate(methods):

        # +1 for initial stage
        results_for_method = np.zeros((steps + 1,))

        for seed in seeds:
            seed_folder = root_path + 'seed' + str(seed) + '/'
            measurements = GetAllMeasurements(seed_folder, method, batch_size)
            max_found_values = GetMaxValues(measurements, batch_size)

            results_for_method = np.add(results_for_method, max_found_values)

        results_for_method = results_for_method / len_seeds
        regrets = [average_model_max - res for res in results_for_method.tolist()]
        result = [method_names[index], regrets]
        results.append(result)
        print result

    PlotData(results=results,  output_file_name=output_filename, isTotalReward=False, isRoad=False)

### Road ###
def GetRoadBeta2Regrets():
    seeds = range(0, 35)
    seeds = list(set(seeds) - set([30]))
    root_path = '../../releaseTests/road/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    RoadRegrets(batch_size, root_path, methods, method_names, seeds)


def GetRoadBeta3Regrets():
    seeds = range(0, 32)
    # seeds = list(set(seeds) - set([27, 31]))
    seeds = list(set(seeds) - set([5]))
    root_path = '../../releaseTests/road/beta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    RoadRegrets(batch_size, root_path, methods, method_names, seeds)


def GetRoadTotalRegrets():
    seeds = range(35)
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4', 'mle_h3', 'qEI', 'pe']
    method_names = ['Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'Anytime H = 4', 'MLE H = 3', 'qEI', 'BUCB-PE']

    output_file = '../../result_graphs/simulated_road_regrets.eps'
    root_path = '../../releaseTests/road/b5-18-log/'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds, output_filename=output_file)


#### Simulated ####

def GetSimulatedTotalRegrets():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/rewards-sAD/'
    methods = ['h1', 'h2', 'h3', 'h4', 'anytime_h3', 'mle_h3', 'qEI']
    method_names = ['H = 1', 'H = 2', 'H = 3', 'H = 4', 'Anytime', 'MLE H = 3', 'qEI']

    output_file = '../../result_graphs/simulated_total_regrets.eps'
    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds, output_filename=output_file)


def GetSimulatedBeta2Regrets():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/simulatedBeta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods =  map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds)


def GetSimulatedBeta3Regrets():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/simulatedBeta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods =  map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds)


if __name__ == "__main__":

    GetRoadTotalRegrets()
    # GetRoadBeta2Regrets()
    # GetRoadBeta3Regrets()
    GetSimulatedTotalRegrets()
    #GetSimulatedBeta2Regrets()
    #GetSimulatedBeta3Regrets()

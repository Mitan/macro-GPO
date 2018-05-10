from StringIO import StringIO
import numpy as np

from src.DatasetUtils import GetAllMeasurements, GetMaxValues
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.enum.PlottingEnum import PlottingMethods
from src.model.DatasetGenerator import DatasetGenerator
from src.plotting.ResultsPlotter import PlotData


def GetRoadResultsForMethod(seeds, batch_size, method, root_path, model_max):
    len_seeds = len(seeds)
    steps = 20 / batch_size
    # +1 for initial stage
    results_for_method = np.zeros((steps + 1,))

    all_measurements = np.zeros((len_seeds, steps + 1))

    for ind, seed in enumerate(seeds):
        seed_folder = root_path + 'seed' + str(seed) + '/'
        measurements = GetAllMeasurements(seed_folder, method, batch_size)
        max_found_values = GetMaxValues(measurements, batch_size)
        assert max_found_values.shape == results_for_method.shape

        # results_for_method = np.add(results_for_method, max_found_values)
        all_measurements[ind, :] = max_found_values

    # results_for_method = results_for_method / len_seeds
    error_bars = np.std(all_measurements, axis=0) / np.sqrt(len_seeds)

    means = np.mean(all_measurements, axis=0)
    regrets = [model_max - res for res in means.tolist()]
    print(error_bars)
    return regrets, error_bars.tolist()
    # result = [method_names[index], regrets]


def RobotRegrets(batch_size, root_path, methods, method_names, seeds, output_filename, plottingType, plot_bars):

    dataset_generator = DatasetGenerator(dataset_type=DatasetEnum.Robot,
                                         dataset_mode=DatasetModeEnum.Load,
                                         time_slot=16, batch_size=batch_size)
    m = dataset_generator.get_dataset_model(root_folder=root_path + 'seed0/', seed=0, ma_treshold=None)

    model_max = m.GetMax()

    results = []

    # for every method
    for index, method in enumerate(methods):
        # todo hack
        adjusted_batch_size = 1 if method == 'ei' else batch_size
        regrets, error_bars = GetRoadResultsForMethod(seeds, adjusted_batch_size, method, root_path, model_max)
        result = [method_names[index], regrets, error_bars]
        results.append(result)
        # print result
    PlotData(results=results, output_file_name=output_filename,
             plottingType=plottingType, dataset='robot',plot_bars=plot_bars)
    return results


def RoadRegrets(batch_size, root_path, methods, method_names, seeds, output_filename, plottingType):
    """
    root_path = '../../releaseTests/road/b5-18-log/'
    seeds = range(35)
    seeds = list(set(seeds) - set([31]))
    batch_size = 5
    methods = ['qEI', 'h1', 'anytime_h2', 'anytime_h3', 'mle_h3']
    method_names = ['qEI', 'Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'MLE H = 3']
    """

    dataset_generator = DatasetGenerator(dataset_type=DatasetEnum.Road,
                                 dataset_mode=DatasetModeEnum.Load,
                                         time_slot=18, batch_size=batch_size)
    m = dataset_generator.get_dataset_model(root_folder=root_path+'seed0/', seed=0, ma_treshold=None)

    model_max = m.GetMax()

    results = []

    # for every method
    for index, method in enumerate(methods):
        # todo hack
        adjusted_batch_size = 1 if method == 'ei' else batch_size
        regrets = GetRoadResultsForMethod(seeds, adjusted_batch_size, method, root_path, model_max)
        result = [method_names[index], regrets]
        results.append(result)
        # print result

    PlotData(results=results, output_file_name=output_filename, plottingType=plottingType, dataset='road')
    return results


def SimulatedRegrets(batch_size, root_path, methods, method_names, seeds, output_filename, plottingType):
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
        seed_dataset_path = root_path + 'seed' + str(seed) + '/'
        dataset_generator = DatasetGenerator(dataset_type=DatasetEnum.Simulated, dataset_mode=DatasetModeEnum.Load,
                                             time_slot=None, batch_size=batch_size)
        m = dataset_generator.get_dataset_model(root_folder=seed_dataset_path, seed=seed, ma_treshold=None)
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
        # print result

    PlotData(results=results, output_file_name=output_filename, plottingType=plottingType, dataset='simulated')
    return results



#### Simulated ####


def GetSimulatedTotalRegrets_H4Samples():
    seeds = range(66, 102)
    batch_size = 4

    root_path = '../../simulated_tests/h4_samples/'
    root_path = '../../releaseTests/simulated/h4_samples/'
    
    methods = ['h4', 'new_new_h4_20', 'h4_5']
    method_names = ['N=100', 'N=20', 'N=5']
    output_file = '../../result_graphs/eps/simulated_h4_samples_simple_regrets.eps'

    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetSimulatedBeta2Regrets():
    seeds = range(66, 102)
    batch_size = 4
    """
    root_path = '../../releaseTests/simulated/simulatedBeta2/'
    root_path = '../../simulated_tests/beta2/'

    beta_list = [ 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)
    
    root_path = '../../simulated_tests/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    """
    root_path = '../../simulated_tests/beta2-good/'

    beta_list = [0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' +  str(2* float(x)), str_beta)

    output_file = '../../result_graphs/eps/simulated/simulated_beta2_simple_regrets.eps'
    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetSimulatedBeta3Regrets():
    seeds = range(66, 102)
    batch_size = 4
    """
    root_path = '../../releaseTests/simulated/simulatedBeta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]

    root_path = '../../simulated_tests/beta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    
    root_path = '../../simulated_tests/beta3_good_zero_mean/'
    beta_list = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)
    """
    root_path = '../../simulated_tests/beta3-good/'
    beta_list = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    # beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + str(2* float(x)), str_beta)

    output_file = '../../result_graphs/eps/simulated/simulated_beta3_simple_regrets.eps'
    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)

#### Robot

if __name__ == "__main__":
    """
    # GetRoadTotalRegrets()
    # GetRoadTotalRegrets_H2Full()
    GetRoadBeta2Regrets()
    GetRoadBeta3Regrets()
    # GetSimulatedBeta2Regrets()
    # GetSimulatedBeta3Regrets()
    # GetRobotTotalRegrets_beta2()
    # GetRobotTotalRegrets_beta3()
   
    GetSimulatedTotalRegrets()
    GetSimulatedTotalRegrets_H4Samples()
    """
    """
    GetRobotTotalRegrets()
    GetRobotTotalRegrets_H2Full()
    GetRobotTotalRegrets_H4Samples()
    """

    """
    GetRoadTotalRegrets_H2Full_H4Samples()
    GetRobotTotalRegrets_H4Samples()
    """
    # GetRobotTotalRegrets()
    # GetRobotTotalRegrets_H2Full()
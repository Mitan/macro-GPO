from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.model.DatasetGenerator import DatasetGenerator
from src.plotting.ResultsPlotter import PlotData
import numpy as np

from src.DatasetUtils import *


def CalculateRoadResultsForOneMethod(batch_size, model_mean, seeds, method, tests_source_path):
    len_seeds = len(seeds)

    steps = 20 / batch_size
    scaled_model_mean = np.array([(1 + batch_size * i) * model_mean for i in range(steps + 1)])
    # print scaled_model_mean
    all_measurements = np.empty([len_seeds, steps+1])

    number_of_location = 0

    # +1 initial point
    results_for_method = np.zeros((steps + 1,))
    counter = 0
    for seed in seeds:

        seed_folder = tests_source_path + 'seed' + str(seed) + '/'
        try:
            # all measurements, unnormalized
            measurements = GetAllMeasurements(seed_folder, method, batch_size)
            number_of_location += 1
        except:
            print seed_folder, method
            continue

        rewards = GetAccumulatedRewards(measurements, batch_size)
        all_measurements[counter, :] = rewards
        results_for_method = np.add(results_for_method, rewards)
        counter+=1

    # check that we collected data for every location
    # print method
    # print number_of_location, len(seeds)
    assert number_of_location == len(seeds)
    # print results_for_method

    results_for_method = results_for_method / len_seeds
    """
    print results_for_method
    print np.mean(all_measurements, axis=0)
    all_measurements  = all_measurements - scaled_model_mean
    print np.std(all_measurements, axis=0)
    """
    scaled_results = results_for_method - scaled_model_mean
    # print method, scaled_results
    # result = [method_names[index], scaled_results.tolist()]
    return scaled_results.tolist()


def RobotRewards(batch_size, tests_source_path, methods, method_names, seeds, output_filename, time_slot, plottingType):
    results = []

    dataset_generator = DatasetGenerator(dataset_type=DatasetEnum.Robot,
                                         dataset_mode=DatasetModeEnum.Load,
                                         time_slot=16, batch_size=batch_size)
    m = dataset_generator.get_dataset_model(root_folder=tests_source_path + 'seed0/', seed=0, ma_treshold=None)
    model_mean = m.mean

    for index, method in enumerate(methods):
        # todo hack
        adjusted_batch_size = 1 if (method == 'ei' or method == 'pi') else batch_size
        scaled_results = CalculateRoadResultsForOneMethod(adjusted_batch_size, model_mean, seeds, method,
                                                          tests_source_path)
        result = [method_names[index], scaled_results]
        results.append(result)

    PlotData(results=results, output_file_name=output_filename, plottingType=plottingType, dataset='robot')
    return results


def RoadRewards(batch_size, tests_source_path, methods, method_names, seeds, output_filename, plottingType):
    """
    seeds = range(36)
    seeds = list(set(seeds) - set([31]))
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'mle_h3', 'qEI']
    method_names = ['Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'MLE H = 3', 'qEI']

    root_path = '../../releaseTests/road/b5-18-log/'
    """
    results = []

    dataset_generator = DatasetGenerator(dataset_type=DatasetEnum.Road,
                                 dataset_mode=DatasetModeEnum.Load,
                                         time_slot=18, batch_size=batch_size)
    m = dataset_generator.get_dataset_model(root_folder=tests_source_path+'seed0/', seed=0, ma_treshold=None)

    model_mean = m.mean

    for index, method in enumerate(methods):
        # todo hack
        adjusted_batch_size = 1 if method == 'ei' else batch_size
        scaled_results = CalculateRoadResultsForOneMethod(adjusted_batch_size, model_mean, seeds, method,
                                                          tests_source_path)
        result = [method_names[index], scaled_results]
        results.append(result)

    PlotData(results=results, output_file_name=output_filename, plottingType=plottingType, dataset='road')
    return results


def SimulatedCumulativeRegrets(batch_size, tests_source_path, methods, method_names, seeds, output_filename, plottingType):
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

    # sum_model_mean = 0
    sum_model_max = 0
    model_max_values = {}

    for seed in seeds:
        seed_dataset_path = tests_source_path + 'seed' + str(seed) + '/'
        dataset_generator = DatasetGenerator(dataset_type=DatasetEnum.Simulated, dataset_mode=DatasetModeEnum.Load,
                                             time_slot=None, batch_size=batch_size)
        m = dataset_generator.get_dataset_model(root_folder=seed_dataset_path, seed=seed, ma_treshold=None)
        # sum_model_mean += m.mean
        global_max = m.GetMax()
        # reachable_max = max(map(lambda x: m(x), reachable_locations))
        # print(global_max - reachable_max)
        # todo nb
        # global_max = reachable_max

        model_max_values[seed] = global_max
        sum_model_max += global_max

    average_model_max = sum_model_max / len_seeds

    # average_model_mean = sum_model_mean / len_seeds

    # print scaled_model_mean

    for index, method in enumerate(methods):
        number_of_location = 0
        adjusted_batch_size = 1 if method == 'rollout_h4_gamma1' or method == 'rollout_h4_gamma1_ei' else batch_size
        steps = 20 / adjusted_batch_size
        # +1 initial point
        # results_for_method = np.zeros((steps + 1,))
        all_regrets = np.zeros((len_seeds, steps + 1))
        for ind, seed in enumerate(seeds):

            seed_folder = tests_source_path + 'seed' + str(seed) + '/'

            # all measurements, unnormalized
            measurements = GetAllMeasurements(seed_folder, method, adjusted_batch_size)
            number_of_location += 1

            # now returns list
            rewards = GetAccumulatedRewards(measurements, adjusted_batch_size)
            # rewards divided by the number of points

            averaged_rewards = [rewards[i] / (1 + adjusted_batch_size* i) for i in range(len(rewards))]
            all_regrets[ind, :] = averaged_rewards
            # results_for_method = np.add(results_for_method, rewards)

        # check that we collected data for every location
        # print method
        # print number_of_location, len(seeds)
        assert number_of_location == len(seeds)
        """
        # print results_for_method
        results_for_method = results_for_method / len_seeds
        scaled_model_mean = np.array([(1 + adjusted_batch_size * i) * average_model_mean for i in range(steps + 1)])
        scaled_results = results_for_method - scaled_model_mean
        result = [method_names[index], scaled_results.tolist(), []]
        results.append(result)
        """
        error_bars = np.std(all_regrets, axis=0) / (np.sqrt(len_seeds))
        # print(error_bars)
        means = np.mean(all_regrets, axis=0)

        # regrets = [average_model_max - res for res in means.tolist()]
        regrets = means.tolist()

        result = [method_names[index], regrets, error_bars.tolist()]
        results.append(result)

    PlotData(results=results, output_file_name=output_filename, dataset='simulated', plottingType=plottingType, plot_bars=True)
    return results


def SimulatedRewards(batch_size, tests_source_path, methods, method_names, seeds, output_filename, plottingType):
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
        seed_dataset_path = tests_source_path + 'seed' + str(seed) + '/'
        dataset_generator = DatasetGenerator(dataset_type=DatasetEnum.Simulated, dataset_mode=DatasetModeEnum.Load,
                                             time_slot=None, batch_size=batch_size)
        m = dataset_generator.get_dataset_model(root_folder=seed_dataset_path, seed=seed, ma_treshold=None)
        sum_model_mean += m.mean

    average_model_mean = sum_model_mean / len_seeds

    # print scaled_model_mean

    for index, method in enumerate(methods):
        number_of_location = 0
        adjusted_batch_size = 1 if method == 'rollout_h4_gamma1' or method == 'rollout_h4_gamma1_ei' else batch_size
        steps = 20 / adjusted_batch_size
        # +1 initial point
        results_for_method = np.zeros((steps + 1,))
        for seed in seeds:

            seed_folder = tests_source_path + 'seed' + str(seed) + '/'

            # all measurements, unnormalized
            measurements = GetAllMeasurements(seed_folder, method, adjusted_batch_size)
            number_of_location += 1

            # now returns list
            rewards = np.array(GetAccumulatedRewards(measurements, adjusted_batch_size))

            results_for_method = np.add(results_for_method, rewards)

        # check that we collected data for every location
        # print method
        # print number_of_location, len(seeds)
        assert number_of_location == len(seeds)
        # print results_for_method
        results_for_method = results_for_method / len_seeds
        scaled_model_mean = np.array([(1 + adjusted_batch_size * i) * average_model_mean for i in range(steps + 1)])
        scaled_results = results_for_method - scaled_model_mean
        result = [method_names[index], scaled_results.tolist(), []]
        results.append(result)

    PlotData(results=results, output_file_name=output_filename, dataset='simulated', plottingType=plottingType)
    return results

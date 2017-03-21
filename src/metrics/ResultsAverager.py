from src.ResultsPlotter import PlotData
import numpy as np

from src.DatasetUtils import GetAllMeasurements, GetAccumulatedRewards, GenerateModelFromFile, GenerateRoadModelFromFile


def RoadRewards(batch_size, tests_source_path, methods, method_names, seeds, output_filename):
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

            seed_folder = tests_source_path + 'seed' + str(seed) + '/'
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

    PlotData(results=results, output_file_name=output_filename, isTotalReward=True, isRoad=True)


def SimulatedRewards(batch_size, tests_source_path, methods, method_names, seeds, output_filename):
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
        seed_dataset_path = tests_source_path + 'seed' + str(seed) + '/dataset.txt'
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

            seed_folder = tests_source_path + 'seed' + str(seed) + '/'
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
    PlotData(results=results, output_file_name=output_filename, isRoad=False, isTotalReward=True)


##### Simulated ####
def GetSimulatedTotalRewards():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/rewards-sAD/'


    methods = ['h1', 'h2', 'h3', 'h4', 'anytime_h3', 'mle_h4', 'qEI', 'pe']

    method_names = [r'$H = 1$', r'$H = 2$', r'$H = 3$', r'$H = 4$', r'$H^* = 3$',  r'MLE $H = 4$', 'qEI', 'BUCB-PE']

    output_file = '../../result_graphs/eps/simulated_total_rewards.eps'

    methods = ['s150_750_anytime_h4', 's200_750_anytime_h4', 's250_750_anytime_h4', 's300_750_anytime_h4',
               's150_1000_anytime_h4', 's200_1000_anytime_h4', 's250_1000_anytime_h4', 's300_1000_anytime_h4',
               's150_1200_anytime_h4', 's200_1200_anytime_h4', 's250_1200_anytime_h4', 's300_1200_anytime_h4',
               's150_1500_anytime_h4', 's200_1500_anytime_h4', 's250_1500_anytime_h4', 's300_1500_anytime_h4',
               's150_2000_anytime_h4', 's200_2000_anytime_h4']
    methods = ['s300_100k_anytime_h4']

    method_names = [r'$150-750$', r'$200-750$', r'$250-750$', r'$300-750$',
                    r'$150-1000$', r'$200-1000$', r'$250-1000$', r'$300-1000$',
                    r'$150-1200$', r'$200-1200$', r'$250-1200$', r'$300-1200$',
                    r'$150-1500$', r'$200-1500$', r'$250-1500$', r'$300-1500$',
                    r'$150-2000$', r'$200-2000$', r'$250-2000$', r'$300-2000$']
    root_path = '../../7anytime/'
    output_file = '../../result_graphs/eps/anytime.eps'

    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file)


def GetSimulatedBeta2Rewards():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/simulatedBeta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/simulated_beta2_rewards.eps'

    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file)


def GetSimulatedBeta3Rewards():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/simulatedBeta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/simulated_beta3_rewards.eps'

    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file)


####### Road ########
def GetRoadBeta2Rewards():
    seeds = range(35)
    root_path = '../../releaseTests/road/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/road_beta2_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file)


def GetRoadBeta3Rewards():
    seeds = range(35)
    root_path = '../../releaseTests/road/beta3/'
    # root_path = '../../last_Beta3/'
    #root_path = '../../zero_last_Beta3/'
    #root_path = '../../copy_beta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/road_beta3_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file)


def GetRoadTotalRewards():
    seeds = range(35)
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4', 'mle_h4', 'qEI', 'pe']
    method_names = [r'$H = 1$', r'$H^* = 2$', r'$H^* = 3$', r'$H^* = 4$', r'MLE $H = 4$', 'qEI', 'BUCB-PE']

    root_path = '../../releaseTests/road/b5-18-log/'

    output_file = '../../result_graphs/eps/road_total_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file)


if __name__ == "__main__":
    """
    GetRoadBeta3Rewards()
    GetRoadBeta2Rewards()
    GetRoadTotalRewards()

    GetSimulatedBeta2Rewards()
    GetSimulatedBeta3Rewards()
    """
    GetSimulatedTotalRewards()

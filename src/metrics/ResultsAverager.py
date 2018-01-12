from src.ResultsPlotter import PlotData
import numpy as np

from src.DatasetUtils import *


def CalculateRoadResultsForOneMethod(batch_size, model_mean, seeds, method, tests_source_path):
    len_seeds = len(seeds)

    steps = 20 / batch_size
    scaled_model_mean = np.array([(1 + batch_size * i) * model_mean for i in range(steps + 1)])
    # print scaled_model_mean

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
    print method, scaled_results
    # result = [method_names[index], scaled_results.tolist()]
    return scaled_results.tolist()


def RobotRewards(batch_size, tests_source_path, methods, method_names, seeds, output_filename, time_slot, isBeta = False):
    results = []

    data_file = '../../datasets/robot/selected_slots/slot_' + str(time_slot) + '/final_slot_' + str(time_slot) + '.txt'
    neighbours_file = '../../datasets/robot/all_neighbours.txt'
    coords_file = '../../datasets/robot/all_coords.txt'
    m = GenerateRobotModelFromFile(data_filename=data_file, coords_filename=coords_file,
                                   neighbours_filename=neighbours_file)
    model_mean = m.mean

    for index, method in enumerate(methods):
        # todo hack
        adjusted_batch_size = 1 if (method == 'ei' or method == 'pi') else batch_size
        scaled_results = CalculateRoadResultsForOneMethod(adjusted_batch_size, model_mean, seeds, method,
                                                          tests_source_path)
        result = [method_names[index], scaled_results]
        results.append(result)

    PlotData(results=results, output_file_name=output_filename, isTotalReward=True, type='robot', isBeta=isBeta)


def RoadRewards(batch_size, tests_source_path, methods, method_names, seeds, output_filename, isBeta = False):
    """
    seeds = range(36)
    seeds = list(set(seeds) - set([31]))
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'mle_h3', 'qEI']
    method_names = ['Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'MLE H = 3', 'qEI']

    root_path = '../../releaseTests/road/b5-18-log/'
    """
    results = []

    dataset_file_name = '../../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(dataset_file_name)
    model_mean = m.mean

    for index, method in enumerate(methods):
        # todo hack
        adjusted_batch_size = 1 if method == 'ei' else batch_size
        scaled_results = CalculateRoadResultsForOneMethod(adjusted_batch_size, model_mean, seeds, method,
                                                          tests_source_path)
        result = [method_names[index], scaled_results]
        results.append(result)

    PlotData(results=results, output_file_name=output_filename, isTotalReward=True,type='road', isBeta=isBeta)


def SimulatedRewards(batch_size, tests_source_path, methods, method_names, seeds, output_filename, isBeta = False):
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
    PlotData(results=results, output_file_name=output_filename, type='simulated', isTotalReward=True, isBeta=isBeta)


##### Simulated ####
def GetSimulatedTotalRewards():
    seeds = range(66, 102)
    batch_size = 4
    """
    root_path = '../../releaseTests/simulated/rewards-sAD/'
    methods = ['h1', 'h2', 'h3', 'h4', '2_s250_100k_anytime_h4', 'mle_h4', 'new_fixed_pe', 'gp-bucb', 'r_qei']

    method_names = ['DB-GP-UCB', r'$\epsilon$-Macro-GPO  $H = 2$', r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI']

    output_file = '../../result_graphs/eps/simulated_total_rewards.eps'

    """
    root_path = '../../simulated_tests/anytime/'
    methods = ['anytime_h4_300']
    method_names = ['Anytime']
    output_file = '../../result_graphs/eps/temp_simulated_total_rewards.eps'

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
                     seeds=seeds, output_filename=output_file, isBeta = True)


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
                     seeds=seeds, output_filename=output_file, isBeta = True)



def GetSimulated_H4Samples_TotalRewards():
    seeds = range(66, 102)
    batch_size = 4

    root_path = '../../simulated_tests/h4_samples/'
    methods = ['h4', 'h4_20', 'h4_5']
    method_names = ['N=100', 'N=20', 'N=5']
    output_file = '../../result_graphs/eps/simulated_h4_samples_total_rewards.eps'

    SimulatedRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                     seeds=seeds, output_filename=output_file)


####### Road ########
def GetRoadBeta2Rewards():
    seeds = range(35)
    # seeds = list(set(range(35))
    root_path = '../../releaseTests/road/beta2/'
    root_path = '../../road_tests/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/road_beta2_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file, isBeta = True)


def GetRoadBeta3Rewards():
    seeds = range(35)
    # seeds = list(set(range(35)) - set([22]))
    root_path = '../../releaseTests/road/beta3/'
    root_path = '../../road_tests/beta3/'
    # root_path = '../../last_Beta3/'
    # root_path = '../../zero_last_Beta3/'
    # root_path = '../../copy_beta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/road_beta3_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file, isBeta = True)


def GetRoad_H4Samples_TotalRewards():
    seeds = range(35)
    batch_size = 5

    methods = ['anytime_h4_5', 'anytime_h4_50', 'anytime_h4']
    methods = ['anytime_h4_5', 'anytime_h4']

    method_names = [r'$N = 5$', r'$N = 50$', r'$N = 300$']
    method_names = [r'$N = 5$', r'$N = 300$']

    root_path = '../../road_tests/new_h4/'

    output_file = '../../result_graphs/eps/road_h4samples_total_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file)

def GetRoadTotalRewards():
    # seeds = list(set(range(35)) - set([22]))
    seeds = range(35)
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4', 'mle_h4','new_ixed_pe', 'bucb', 'r_qei']
    methods = ['anytime_h2', 'anytime_h3']
    # methods = ['anytime_h4_5']

    method_names = ['DB-GP-UCB', r'Anytime-$\epsilon$-Macro-GPO  $H = 2$', r'Anytime-$\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB', r'$q$-EI']

    method_names = [ "H = 2", "H = 3"]

    root_path = '../../releaseTests/road/b5-18-log/'
    root_path = '../../road_tests/tests1/'

    output_file = '../../result_graphs/eps/road_total_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file)


def GetRoad_H2Full_TotalRewards():
    seeds = range(35)
    # seeds = list(set(range(35)) - set([22]))
    batch_size = 5

    # methods = ['anytime_h2_full', 'anytime_h2', 'anytime_h4']
    methods = ['anytime_h2_full_2121', 'anytime_h2', 'anytime_h4', 'ei']
    method_names = [ r'Anytime-$\epsilon$-Macro-GPO  $H = 2$ (all MA)',
                     r'Anytime-$\epsilon$-Macro-GPO  $H = 2$  (selected MA)',
                     r'Anytime-$\epsilon$-Macro-GPO  $H = 4$  (selected MA)',
                     'EI (all MA)']

    root_path = '../../releaseTests/road/tests2full/'

    output_file = '../../result_graphs/eps/road_h2_full_total_rewards.eps'

    RoadRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                seeds=seeds, output_filename=output_file)


##### Robot #######

def GetRobotBeta2Rewards():
    seeds = range(35)
    seeds = range(35)

    root_path = '../../releaseTests/robot/beta_new2/'
    root_path = '../../robot_tests/beta2/'
    # root_path = '../../robot_tests/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    batch_size = 5
    time_slot = 16

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/robot_beta2_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot, isBeta = True)


def GetRobotBeta3Rewards():
    seeds = range(35)
    # seeds = list(set(range(35)) - set([33, 34]))
    root_path = '../../releaseTests/robot/beta_new3/'
    root_path = '../../robot_tests/beta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0,  5.0]

    beta_list = [0.0, 0.05, 0.1]

    batch_size = 5
    time_slot = 16

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/robot_beta3_rewards.eps'
    output_file = '../../result_graphs/eps/new_robot_beta3_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot, isBeta = True)


def GetRobot_H4Samples_TotalRewards():
    seeds = range(35)
    batch_size = 5

    time_slot = 16

    methods = ['new_anytime_h4_ 5', 'anytime_h4_ 50', 'anytime_h4']
    methods = ['new_anytime_h4_ 5', 'anytime_h4_5','anytime_h4']

    method_names = [r'$N = 5$', r'$N = 50$', r'$N = 300$']
    method_names = [r'$N = 5$', r'$N = 5$a' r'$N = 300$']

    root_path = '../../robot_tests/h4_samples/'

    output_file = '../../result_graphs/eps/robot_h4samples_total_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot)

def GetRobotTotalRewards():
    seeds = range(35)
    batch_size = 5

    time_slot = 16

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4', 'mle_h4', 'r_qei',  'fixed_pe', 'gp-bucb']
    # methods = ['new_anytime_h1', 'new_anytime_h2', 'new_anytime_h3']

    method_names = [r'$H = 1$', r'$H^* = 2$', r'$H^* = 3$', r'$H^* = 4$', r'MLE $H = 4$', 'qEI', 'GP-BUCB-PE', 'GP-BUCB']
    method_names = ['DB-GP-UCB', r'Anytime-$\epsilon$-Macro-GPO  $H = 2$', r'Anytime-$\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', r'$q$-EI', 'GP-UCB-PE', 'GP-BUCB']
    # method_names = [r'$H = 1$', r'$H^* = 2$', r'$H^* = 3$']
    root_path = '../../releaseTests/robot/slot_16/'
    # root_path = '../../robot_tests/tests3/'

    output_file = '../../result_graphs/eps/robot_total_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot)


def GetRobot_H2Full_TotalRewards():
    seeds = range(35)
    batch_size = 5
    time_slot = 16

    # methods = ['anytime_h2_full', 'anytime_h2', 'anytime_h4']
    methods = ['anytime_h2_full_2121', 'anytime_h2', 'anytime_h4', 'ei']
    method_names = [r'$H^* = 2$ (all MA)', r'$H^* = 2$ (selected MA)', r'$H^* = 4$ (selected MA)']
    method_names = [r'Anytime-$\epsilon$-Macro-GPO  $H = 2$ (all MA)',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 2$  (selected MA)',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$  (selected MA)',
                    'EI (all MA)']

    root_path = '../../releaseTests/robot/h2_full/'
    # root_path = '../../robot_tests/21_full/'

    output_file = '../../result_graphs/eps/robot_h2_full_total_rewards.eps'

    RobotRewards(batch_size=batch_size, tests_source_path=root_path, methods=methods, method_names=method_names,
                 seeds=seeds, output_filename=output_file, time_slot=time_slot)


if __name__ == "__main__":
    """
    GetRoadTotalRewards()
    GetRoadBeta3Rewards()
    GetRoadBeta2Rewards()
    GetRoad_H2Full_TotalRewards()
    
    GetSimulatedBeta2Rewards()
    GetSimulatedBeta3Rewards()
    GetSimulatedTotalRewards()
    
    GetRobotTotalRewards()
    GetRobotBeta2Rewards()
    GetRobotBeta3Rewards()
    GetRobot_H2Full_TotalRewards()
    """
    # GetRobotTotalRewards()
    # GetRoad_H4Samples_TotalRewards()
    GetRobot_H4Samples_TotalRewards()
    # GetSimulatedTotalRewards()
    # GetRoadBeta3Rewards()
    # GetSimulatedTotalRewards()
    # GetRoadBeta2Rewards()
    # GetRoadBeta3Rewards()
    # GetSimulated_H4Samples_TotalRewards()

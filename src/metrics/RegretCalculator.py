from StringIO import StringIO
import numpy as np

from src.DatasetUtils import GenerateRoadModelFromFile, GetAllMeasurements, GetMaxValues, GenerateModelFromFile, \
    GenerateRobotModelFromFile
from src.PlottingEnum import PlottingMethods
from src.ResultsPlotter import PlotData


def GetRoadResultsForMethod(seeds, batch_size, method, root_path, model_max):
    len_seeds = len(seeds)
    steps = 20 / batch_size
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
    return regrets
    # result = [method_names[index], regrets]


def RobotRegrets(batch_size, root_path, methods, method_names, seeds, output_filename, plottingType):
    time_slot = 16
    data_file = '../../datasets/robot/selected_slots/slot_' + str(time_slot) + '/noise_final_slot_' + str(time_slot) + '.txt'
    neighbours_file = '../../datasets/robot/all_neighbours.txt'
    coords_file = '../../datasets/robot/all_coords.txt'
    m = GenerateRobotModelFromFile(data_filename=data_file, coords_filename=coords_file,
                                   neighbours_filename=neighbours_file)
    model_max = m.GetMax()

    results = []

    # for every method
    for index, method in enumerate(methods):
        # todo hack
        adjusted_batch_size = 1 if method == 'ei' else batch_size
        regrets = GetRoadResultsForMethod(seeds, adjusted_batch_size, method, root_path, model_max)
        result = [method_names[index], regrets]
        results.append(result)
        print result

    PlotData(results=results, output_file_name=output_filename, plottingType=plottingType, dataset='robot')


def RoadRegrets(batch_size, root_path, methods, method_names, seeds, output_filename, plottingType):
    """
    root_path = '../../releaseTests/road/b5-18-log/'
    seeds = range(35)
    seeds = list(set(seeds) - set([31]))
    batch_size = 5
    methods = ['qEI', 'h1', 'anytime_h2', 'anytime_h3', 'mle_h3']
    method_names = ['qEI', 'Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'MLE H = 3']
    """


    dataset_file_name = '../../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(dataset_file_name)
    model_max = m.GetMax()

    results = []

    # for every method
    for index, method in enumerate(methods):
        # todo hack
        adjusted_batch_size = 1 if method == 'ei' else batch_size
        regrets = GetRoadResultsForMethod(seeds, adjusted_batch_size, method, root_path, model_max)
        result = [method_names[index], regrets]
        results.append(result)
        print result

    PlotData(results=results, output_file_name=output_filename, plottingType=plottingType, dataset='road')


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

    PlotData(results=results, output_file_name=output_filename, plottingType=plottingType, dataset='simulated')

### Road ###
def GetRoadBeta2Regrets():
    seeds = range(0, 35)
    # seeds = list(set(seeds) - set([27, 31]))
    # seeds = list(set(seeds) - set([5]))
    root_path = '../../road_tests/beta2/'
    root_path = '../../releaseTests/road/beta2/'

    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    # methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/road_beta2_regrets.eps'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRoadBeta3Regrets():
    seeds = range(0, 35)
    # seeds = list(set(seeds) - set([27, 31]))
    # seeds = list(set(seeds) - set([5]))
    root_path = '../../road_tests/beta3/'

    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = ['anytime_h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/road_beta3_regrets.eps'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRoadTotalRegrets():
    seeds = range(35)
    batch_size = 5

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4', 'mle_h4', 'new_ixed_pe', 'bucb', 'r_qei']

    method_names = [r'$H = 1$', r'$H^* = 2$', r'$H^* = 3$', r'$H^* = 4$', r'MLE $H = 4$', 'GP-BUCB-PE', 'GP-BUCB',
                    'qEI']

    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4_300', 'mle_h4', 'new_ixed_pe', 'bucb', 'r_qei']
    # methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'anytime_h4_300']

    method_names = ['DB-GP-UCB', r'Anytime-$\epsilon$-Macro-GPO  $H = 2$', r'Anytime-$\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB', r'$q$-EI']



    # root_path = '../../road_tests/tests1/'
    output_file = '../../result_graphs/eps/road_simple_regrets.eps'
    root_path = '../../releaseTests/road/b5-18-log-NEW-copy/'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRoadTotalRegrets_H2Full():
    seeds = range(35)
    batch_size = 5

    methods = ['anytime_h2_full_2', 'anytime_h2', 'anytime_h4_300', 'ei']
    # methods = ['anytime_h2_full_2', 'anytime_h2', 'anytime_h4_300']
    method_names = [ r'Anytime-$\epsilon$-Macro-GPO  $H = 2$ (all MA)',
                     r'Anytime-$\epsilon$-Macro-GPO  $H = 2$  (selected MA)',
                     r'Anytime-$\epsilon$-Macro-GPO  $H = 4$  (selected MA)',
                     'EI (all MA)']

    output_file = '../../result_graphs/eps/temp_road_h2_full_simple_regrets.eps'
    root_path = '../../releaseTests/road/tests2full/'
    root_path = '../../releaseTests/road/tests2full-NEW/'

    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRoadTotalRegrets_H2Full_H4Samples():
    seeds = range(35)
    batch_size = 5

    methods = ['anytime_h4_5', 'anytime_h4_50', 'anytime_h4']
    methods = ['anytime_h4_5', 'anytime_h4']

    method_names = [r'$N = 5$', r'$N = 50$', r'$N = 300$']
    method_names = [r'$N = 5$', r'$N = 300$']

    root_path = '../../road_tests/new_h4/'

    output_file = '../../result_graphs/eps/road_h4samples_simple_regrets.eps'

    RoadRegrets(batch_size, root_path, methods, method_names, seeds, output_filename=output_file)



def GetRoadBeta2Regrets():
    """
    seeds = range(35)
    root_path = '../../releaseTests/road/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    batch_size = 5

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)
    RoadRegrets(batch_size, root_path, methods, method_names, seeds)
    """
    seeds = range(35)
    batch_size = 5

    root_path = '../../road_tests/beta2/'
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h2'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)

    root_path = '../../releaseTests/road/beta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/road_beta2_regrets.eps'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRoadBeta3Regrets():
    seeds = range(35)
    batch_size = 5

    root_path = '../../road_tests/beta3/'
    beta_list = [0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = ['anytime_h3'] + map(lambda x: 'beta' + x, str_beta)
    method_names = ['beta = 0.0'] + map(lambda x: 'beta = ' + x, str_beta)

    root_path = '../../releaseTests/road/beta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/road_beta3_regrets.eps'
    RoadRegrets(batch_size, root_path, methods, method_names, seeds,
                output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


#### Simulated ####

def GetSimulatedTotalRegrets():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/rewards-sAD/'

    methods = ['h1', 'h2', 'h3', 'h4', '2_s250_100k_anytime_h4', 'mle_h4', 'new_fixed_pe', 'gp-bucb', 'r_qei']

    method_names = ['DB-GP-UCB', r'$\epsilon$-Macro-GPO  $H = 2$', r'$\epsilon$-Macro-GPO  $H = 3$',
                    r'$\epsilon$-Macro-GPO  $H = 4$',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', 'GP-UCB-PE', 'GP-BUCB',
                    r'$q$-EI']

    output_file = '../../result_graphs/eps/simulated_simple_regrets.eps'
    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)

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
    root_path = '../../releaseTests/simulated/simulatedBeta2/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/simulated_beta2_simple_regrets.eps'
    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetSimulatedBeta3Regrets():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../../releaseTests/simulated/simulatedBeta3/'
    beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    beta_list = [0.0, 0.1, 1.0, 2.0, 5.0]
    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/simulated_beta3_simple_regrets.eps'
    SimulatedRegrets(batch_size, root_path, methods, method_names, seeds,
                     output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)

#### Robot

def GetRobotTotalRegrets():
    seeds = range(35)
    batch_size = 5

    """
    methods = ['h1', 'anytime_h2', 'anytime_h3', 'anytime_h4', 'mle_h4', 'r_qei', 'fixed_pe', 'gp-bucb']

    method_names = ['DB-GP-UCB', r'Anytime-$\epsilon$-Macro-GPO  $H = 2$', r'Anytime-$\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', r'$q$-EI', 'GP-UCB-PE', 'GP-BUCB']

    root_path = '../../releaseTests/robot/slot_16/'
    """
    methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'new_anytime_h4_300', 'mle_h4', 'r_qei', 'pe', 'gp-bucb']

    method_names = ['DB-GP-UCB', r'Anytime-$\epsilon$-Macro-GPO  $H = 2$', r'Anytime-$\epsilon$-Macro-GPO  $H = 3$',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$', r'MLE $H = 4$', r'$q$-EI', 'GP-UCB-PE', 'GP-BUCB']
    root_path = '../../noise_robot_tests/all_tests/'

    output_file = '../../result_graphs/eps/noise_robot_simple_regrets.eps'

    RobotRegrets(batch_size, root_path, methods, method_names, seeds,
                 output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRobotTotalRegrets_H2Full():
    seeds = range(35)
    batch_size = 5
    time_slot = 16

    methods = ['anytime_h2_full_2121', 'anytime_h2', 'anytime_h4', 'ei']
    method_names = [r'Anytime-$\epsilon$-Macro-GPO  $H = 2$ (all MA)',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 2$  (selected MA)',
                    r'Anytime-$\epsilon$-Macro-GPO  $H = 4$  (selected MA)',
                    'EI (all MA)']

    root_path = '../../releaseTests/robot/h2_full/'

    output_file = '../../result_graphs/eps/noise_robot_h2_full_simple_regrets.eps'

    RobotRegrets(batch_size, root_path, methods, method_names, seeds,
                 output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRobotTotalRegrets_H4Samples():
    seeds = range(35)
    batch_size = 5

    time_slot = 16
    """
    methods = ['new_anytime_h4_ 5', 'anytime_h4_ 50', 'anytime_h4']
    methods = ['new_anytime_h4_ 5', 'anytime_h4_5','anytime_h4']
    methods = ['new_anytime_h4_ 5','anytime_h4']

    method_names = [r'$N = 5$', r'$N = 50$', r'$N = 300$']
    method_names = [r'$N = 5$', r'$N = 5$a' r'$N = 300$']
    method_names = [r'$N = 5$',  r'$N = 300$']

    root_path = '../../robot_tests/h4_samples/'
    """
    methods = ['new_anytime_h4_5', 'new_anytime_h4_50', 'new_anytime_h4_300']

    method_names = [r'$N = 5$', r'$N = 50$', r'$N = 300$']

    root_path = '../../noise_robot_tests/h4_tests/'
    output_file = '../../result_graphs/eps/noise_robot_h4samples_simple_regrets.eps'

    RobotRegrets(batch_size, root_path, methods, method_names, seeds,
                 output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)

def GetRobotTotalRegrets_beta2():
    seeds = range(35)
    batch_size = 5

    time_slot = 16
    root_path = '../../noise_robot_tests/beta2_fixed/'
    # root_path = '../../noise_robot_tests/beta2_fixed/'
    # beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.5, 1.0, 2.0, 5.0]

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/noise_robot_beta2_simple_regrets.eps'

    RobotRegrets(batch_size, root_path, methods, method_names, seeds,
                 output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


def GetRobotTotalRegrets_beta3():
    seeds = range(35)
    batch_size = 5

    time_slot = 16
    root_path = '../../noise_robot_tests/beta3_fixed_exp/'
    # beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    beta_list = [0.0, 0.5, 1.0, 2.0, 5.0]

    str_beta = map(str, beta_list)
    methods = map(lambda x: 'beta' + x, str_beta)
    method_names = map(lambda x: 'beta = ' + x, str_beta)

    output_file = '../../result_graphs/eps/additional/noise_robot_beta3_simple_regrets.eps'

    RobotRegrets(batch_size, root_path, methods, method_names, seeds,
                 output_filename=output_file, plottingType=PlottingMethods.SimpleRegret)


if __name__ == "__main__":

    # GetRoadTotalRegrets()
    # GetRoadTotalRegrets_H2Full()
    GetRoadBeta2Regrets()
    GetRoadBeta3Regrets()
    # GetSimulatedBeta2Regrets()
    # GetSimulatedBeta3Regrets()
    # GetRobotTotalRegrets_beta2()
    # GetRobotTotalRegrets_beta3()
    """
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
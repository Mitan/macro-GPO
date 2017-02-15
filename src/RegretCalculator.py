from StringIO import StringIO
import numpy as np

from src.DatasetUtils import GenerateRoadModelFromFile, GetAllMeasurements, GetMaxValues, GenerateModelFromFile
from src.ResultsPlotter import PlotData


def CalculateRoadRegret():
    # cannot use - cylcic linking
    root_path = '../releaseTests/road/b5-18-log/'
    seeds = range(35)
    seeds = list(set(seeds) - set([31]))
    batch_size = 5
    methods = ['qEI', 'h1', 'anytime_h2', 'anytime_h3', 'mle_h3']
    method_names = ['qEI', 'Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'MLE H = 3']

    dataset_file_name = '../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(dataset_file_name)
    model_max = m.GetMax()
    len_seeds = len(seeds)
    steps = 20 / batch_size

    results = []

    # for every method
    for index, method in enumerate(methods):

        # +1 for initial stage
        results_for_method = np.zeros((steps + 1,))

        for seed in seeds:
            seed_folder = root_path + 'seed' + str(seed) + '/'
            measurements =  GetAllMeasurements(seed_folder, method, batch_size)
            max_found_values = GetMaxValues(measurements, batch_size)

            results_for_method = np.add(results_for_method, max_found_values)

        results_for_method = results_for_method / len_seeds
        regrets = [model_max - res for res in results_for_method.tolist()]
        result = [method_names[index], regrets]
        results.append(result)
        print result

    PlotData(results=results, folder_name=root_path, file_name='regrets.png',  isTotalReward=False)


def CalculateSimulatedRegret():
    seeds = range(66, 102)
    batch_size = 4
    root_path = '../releaseTests/simulated/rewards-sAD/'
    methods = ['h1', 'h2', 'h3', 'h4', 'anytime_h3', 'mle_h3', 'qEI']
    method_names = ['H = 1', 'H = 2', 'H = 3', 'H = 4', 'Anytime', 'MLE H = 3', 'qEI']

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

    PlotData(results=results, folder_name=root_path, file_name='regrets.png', isTotalReward=False)

#todo UNUSED

def CalculateAverageRegret(model_max, root_path, seeds, methods, method_names, batch_size):
    # seeds = range(36)

    len_seeds = len(seeds)

    # methods = ['h1', 'anytime_h2', 'anytime_h3','mle_h3', 'qEI' ]
    # method_names = ['Myopic UCB', 'Anytime H = 2', 'Anytime H = 3','MLE H = 3', 'qEI']

    # root_path = '../testsRoad/b' + str(batch_size) + '/'+ str(slot) + '/'

    steps = 20 / batch_size

    results = []

    for index, method in enumerate(methods):

        results_for_method = np.zeros((steps,))

        for seed in seeds:
            seed_folder = root_path + 'seed' + str(seed) + '/'
            max_found_values = CalculateMethodMaxValues(seed_folder, method, batch_size)
            assert max_found_values is not None
            results_for_method = np.add(results_for_method, max_found_values)

        results_for_method = results_for_method / len_seeds
        regrets = [model_max - res for res in results_for_method.tolist()]
        result = [method_names[index], regrets]
        results.append(result)
        print result

    PlotData(results=results, folder_name=root_path, file_name='regrets.png', add_zero=False)


#todo UNUSED
# return list of 1 + number_of_steps values (first is initial value)
def CalculateMethodMaxValues(root_folder, method_name, batch_size):
    n_steps = 20 / batch_size
    max_found_values = []
    for i in range(n_steps):
        step_file_name = root_folder + method_name + '/step' + str(i) + '.txt'
        lines = open(step_file_name).readlines()
        first_line_index = 1 + batch_size + 1 + (1 + batch_size * (i + 1)) + 1
        last_line_index = -1
        stripped_lines = map(lambda x: x.strip(), lines[first_line_index: last_line_index])
        joined_lines = " ".join(stripped_lines)
        assert joined_lines[0] == '['
        assert joined_lines[-1] == ']'
        a = StringIO(joined_lines[1:-1])

        # all measurements obtained by the robot till that step
        measurements = np.genfromtxt(a)

        assert measurements.shape[0] == batch_size * (i + 1) + 1
        # assert we parsed them all as numbers
        assert not np.isnan(measurements).any()

        max_found = max(measurements)
        max_found_values.append(max_found)

    # now measurement var stores all measurements obtained
    initial_value = measurements[0:]
    # add it at the begining
    max_found_values = [initial_value] + max_found_values
    return max_found_values


if __name__ == "__main__":
    """
    # cannot use - cylcic linking
    folder_name = '../testsRoad/b5/18/'
    seeds = range(20)
    b = 5
    methods = ['qEI', 'h1', 'anytime_h2', 'anytime_h3', 'mle_h3']
    method_names = [ 'qEI', 'Myopic UCB', 'Anytime H = 2', 'Anytime H = 3', 'MLE H = 3']

    file_name = '../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(file_name)
    model_max = m.GetMax()
    CalculateAverageRegret(model_max=model_max, root_path=folder_name, seeds=seeds, methods=methods, method_names=method_names,
                           batch_size=b)
    """
    # CalculateRoadRegret()
    CalculateSimulatedRegret()
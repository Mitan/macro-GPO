from StringIO import StringIO
import numpy as np

from src.DatasetUtils import GenerateRoadModelFromFile
from src.ResultsPlotter import PlotData


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

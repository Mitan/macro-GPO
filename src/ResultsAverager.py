from ResultsPlotter import PlotData
import numpy as np
from StringIO import StringIO
from DatasetUtils import GetGCoefficient

seeds = range(66, 102)

# seeds = list(set(seeds) - set([92]))
# seeds = list(set(seeds) - set([88]))

root_path = '../tests/b4_sAD_loc0_h3/'

methods = ['h1', 'h2', 'h3','h4', 'anytime_h3','mle_h3', 'qEI' ]
method_names = ['H = 1', 'H = 2', 'H = 3', 'H = 4', 'Anytime','MLE H = 3', 'qEI']

# methods = ['h1', 'h2', 'h3', 'anytime_h3', 'mle_h3', 'qEI']
# method_names = ['H = 1', 'H = 2', 'H = 3', 'Anytime', 'MLE H = 3', 'qEI']

steps = 5

results = []

coeff_file = open(root_path + 'g_coefficients.txt', 'w')

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
            a = (open(file_path).readlines()[-1])[27: -2]
            # print a
            # print file_path
            number_of_location += 1
        except:
            continue

        a = StringIO(a)

        rewards = np.genfromtxt(a, delimiter=",")
        results_for_method = np.add(results_for_method, rewards)

        g_coefficient = GetGCoefficient(seed_folder, method)
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
    coeff_file.write(method_names[index] + ' ' + str(mean_coefficient) + '\n')

PlotData(results, root_path)
coeff_file.close()

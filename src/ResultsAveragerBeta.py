from ResultsPlotter import PlotData
import numpy as np
from StringIO import StringIO


seeds = range(0, 35)
seeds = list(set(seeds) - set([30]))

seeds = range(16)

root_path = '../releaseTests/road/beta2/'
root_path = '../testsRoadBeta3/b5/18/'
beta_list = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


steps = 4

results = []

for beta in beta_list:
    number_of_location = 0
    results_for_method = np.zeros((steps,))
    for seed in seeds:
        # counter

        file_path = root_path + 'seed' + str(seed) + '/beta' + str(beta) + '/summary.txt'
        try:
            a = (open(file_path).readlines()[-1])[27: -2]
            # print file_path
            # print a
            number_of_location += 1
        except:
            print file_path
            continue

        a = StringIO(a)
        rewards = np.genfromtxt(a, delimiter=",")
        results_for_method = np.add(results_for_method, rewards)
    # check that we collected data for every location
    assert number_of_location == len(seeds), "%s %s" % (number_of_location, len(seeds))
    # print number_of_location, len(seeds),
    # print results_for_method
    results_for_method = results_for_method / number_of_location

    result = ["beta = " + str(beta), results_for_method.tolist()]
    results.append(result)

PlotData(results, root_path)

from ResultsPlotter import PlotData
import numpy as np
from StringIO import StringIO


seeds = range(32)
seeds = list(set(seeds) - set([1]))

root_path = '../testsRoadBeta2/b5/18/'
# root_path = '../testBeta2/'


beta_list = ['1e-05', '1e-06', '1e-07',10**-4, 10**-3, 5* 10**-3, 10**-2, 5* 10**-2,0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]
# beta_list = [10**-3,0.0, 0.1, 1.0, 2.0, 10.0]
beta_list = [10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 5 * 10 ** -3, 10 ** -2, 5 * 10 ** -2, 0.0, 0.1, 1.0,
             2.0, 5.0, 10.0, 50.0, 100.0, 200.0]

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

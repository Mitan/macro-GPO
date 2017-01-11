from ResultsPlotter import PlotData
import numpy as np
from StringIO import StringIO


seeds = range(66,102)

# seeds = list(set(seeds) - set([92]))
root_path = '../testBeta2/'

#methods = ['h1', 'h2', 'h3','h4', 'anytime_h3','mle_h3', 'qEI' ]
#method_names = ['H = 1', 'H = 2', 'H = 3', 'H = 4', 'Anytime','MLE H = 3', 'qEI']

beta_list = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]

methods = ['h1', 'h2', 'h3', 'anytime_h3','mle_h3', 'qEI' ]
method_names = ['H = 1', 'H = 2', 'H = 3', 'Anytime','MLE H = 3', 'qEI']

steps = 5

results = []

for beta in beta_list:
    number_of_location = 0
    results_for_method = np.zeros((steps,))
    for seed in seeds:
        # counter

        file_path = root_path + 'seed' + str(seed) + '/beta' + str(beta) + '/summary.txt'
        # print file_path
        try:
            a = (open(file_path).readlines()[-1])[27: -2]
            # print a
            number_of_location += 1
        except:
            continue

        a = StringIO(a)
        rewards = np.genfromtxt(a, delimiter=",")
        results_for_method = np.add(results_for_method, rewards)
    # check that we collected data for every location
    assert number_of_location == len(seeds)
    #print results_for_method
    results_for_method = results_for_method / number_of_location

    result = ["beta = " + str(beta), results_for_method.tolist()]
    results.append(result)

PlotData(results, root_path)

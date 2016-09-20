from ResultsPlotter import PlotData
import numpy as np
from StringIO import StringIO


seeds = range(0, 15)
root_path = './testsAAAI/'

methods = ['h1', 'h2', 'h3' ]
method_names = ['H = 1', 'H = 2', 'H = 3', 'H = 4']

steps = 20

results = []

for index, method in enumerate(methods):
    number_of_location = 0
    results_for_method = np.zeros((steps,))
    for seed in seeds:
        # counter

        file_path = root_path + 'seed' + str(seed) + '/' + method + '/summary.txt'
        try:
            a = (open(file_path).readlines()[0])[1: -2]
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

    result = [method_names[index], results_for_method.tolist()]
    results.append(result)

PlotData(results, root_path)

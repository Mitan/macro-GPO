from src.ResultsPlotter import PlotData
import numpy as np
from StringIO import StringIO

from src.DatasetUtils import GetAllMeasurements, GetAccumulatedRewards, GenerateModelFromFile, GenerateRoadModelFromFile


def CalculateExpandedNodes(root_path, methods, method_names, seeds):

    nodes_file = open(root_path + 'anytime_expanded_roads.txt', 'w')

    for index, method in enumerate(methods):
        number_of_location = 0

        results_for_method = 0.0
        for seed in seeds:

            seed_folder = root_path + 'seed' + str(seed) + '/'
            file_path = seed_folder + method + '/summary.txt'
            try:
                # normalized
                a = (open(file_path).readlines()[-3])
                number_of_location += 1
            except:
                print file_path
                continue

            parts = a.split()
            current_expanded_nodes = int(parts[4])
            results_for_method += current_expanded_nodes

        assert number_of_location == len(seeds)
        # print results_for_method
        results_for_method = results_for_method / number_of_location
        nodes_file.write(method_names[index] + ' ' + str(results_for_method) + '\n')
    nodes_file.close()



def ExpandedNodesSimulated():
    seeds = range(66, 102)
    root_path = '../../releaseTests/simulated/rewards-sAD/'
    methods = ['anytime_h3']
    method_names = ['Anytime 3']
    CalculateExpandedNodes(root_path, methods, method_names, seeds)


def ExpandedNodesRoads():
    seeds = range(36)
    seeds = list(set(seeds) - set([31]))

    methods = ['anytime_h2', 'anytime_h3']
    method_names = ['Anytime H = 2', 'Anytime H = 3']

    root_path = '../../releaseTests/road/b5-18-log/'
    CalculateExpandedNodes(root_path, methods, method_names, seeds)


if __name__ == "__main__":
    ExpandedNodesRoads()
    ExpandedNodesSimulated()

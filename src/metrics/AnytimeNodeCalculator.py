from src.ResultsPlotter import PlotData
import numpy as np
from StringIO import StringIO

from src.DatasetUtils import GetAllMeasurements, GetAccumulatedRewards, GenerateModelFromFile, \
    GenerateRoadModelFromFile, GenerateRobotModelFromFile


def CalculateExpandedNodes(root_path, methods, method_names, seeds, output_file):
    nodes_file = open(output_file, 'w')

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


def CountExpandedNodesForSingleSeed(m, seed, path):
    seed_folder = path + 'seed' + str(seed) + '/'
    seed_file = seed_folder + 'h1/summary.txt'

    all_summary_lines = open(seed_file).readlines()
    location_lines = all_summary_lines[7: 28]
    # for line in location_lines:
    expanded_nodes = 0
    for i in range(0, 20, 5):
        string_numbers = location_lines[i].replace(',', ' ').replace('[', ' ').replace(']', ' ').split()
        loc = map(float, string_numbers)
        # print loc, len(m.GetSelectedMacroActions(loc))
        expanded_nodes += len(m.GetSelectedMacroActions(loc))
    return expanded_nodes


##### Simulated

def ExpandedNodesSimulated():
    seeds = range(66, 102)
    root_path = '../../releaseTests/simulated/rewards-sAD/'
    methods = ['2_s250_100k_anytime_h4']
    method_names = ['Anytime 4']
    output_file = '../../result_graphs/nodes_simulated.txt'
    CalculateExpandedNodes(root_path, methods, method_names, seeds, output_file=output_file)


# Roads

def ExpandedNodesRoads():
    seeds = range(35)

    methods = ['anytime_h2', 'anytime_h3', 'anytime_h4']
    method_names = ['Anytime H = 2', 'Anytime H = 3', 'Anytime H = 4']

    root_path = '../../releaseTests/road/b5-18-log/'
    output_file = '../../result_graphs/nodes_roads.txt'
    CalculateExpandedNodes(root_path, methods, method_names, seeds, output_file=output_file)


def ExpandedNodesRoads_H2Full():
    seeds = range(35)

    methods = ['anytime_h2_full_2121', 'anytime_h2', 'anytime_h4']
    method_names = ['Anytime H = 2 Full', 'Anytime H = 2', 'Anytime H = 4']

    root_path = '../../releaseTests/road/tests2full/'
    output_file = '../../result_graphs/nodes_roads_h2_full.txt'
    CalculateExpandedNodes(root_path, methods, method_names, seeds, output_file=output_file)


def Road_ExpandedNodesForH1():
    seeds = range(35)
    total_expanded_nodes = 0

    filename = '../../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(filename)
    folder_path = '../../releaseTests/road/b5-18-log/'
    for seed in seeds:
        seed_folder = '../../releaseTests/road/b5-18-log/seed' + str(seed) + '/'
        m.LoadSelectedMacroactions(folder_name=seed_folder, batch_size=5)
        current = CountExpandedNodesForSingleSeed(m=m, seed=seed, path=folder_path)
        print current
        total_expanded_nodes += current

    average_nodes = float(total_expanded_nodes) / len(seeds)
    print average_nodes
    output_file = '../../result_graphs/nodes_road_h1.txt'
    nodes_file = open(output_file, 'w')
    nodes_file.write(str(average_nodes))
    nodes_file.close()


# Robot


def Robot_ExpandedNodesForH1():
    seeds = range(35)
    total_expanded_nodes = 0

    time_slot = 16
    data_filename = '../../datasets/robot/selected_slots/slot_' + str(time_slot) + '/final_slot_' + str(
        time_slot) + '.txt'
    neighbours_filename = '../../datasets/robot/all_neighbours.txt'
    coords_filename = '../../datasets/robot/all_coords.txt'

    m = GenerateRobotModelFromFile(data_filename=data_filename, coords_filename=coords_filename,
                                   neighbours_filename=neighbours_filename)
    folder_path = '../../robot_tests/tests1_16_ok/'
    for seed in seeds:
        seed_folder = folder_path + 'seed' + str(seed) + '/'
        m.LoadSelectedMacroactions(folder_name=seed_folder, batch_size=5)
        current = CountExpandedNodesForSingleSeed(m=m, seed=seed, path=folder_path)
        print current
        total_expanded_nodes += current

    average_nodes = float(total_expanded_nodes) / len(seeds)
    print average_nodes
    output_file = '../../result_graphs/robot_nodes_road_h1.txt'
    nodes_file = open(output_file, 'w')
    nodes_file.write(str(average_nodes))
    nodes_file.close()


def Robot_ExpandedNodes():
    seeds = range(35)

    methods = ['anytime_h2', 'anytime_h3', 'anytime_h4']
    method_names = ['Anytime H = 2', 'Anytime H = 3', 'Anytime H = 4']

    root_path = '../../robot_tests/tests1_16_ok/'
    output_file = '../../result_graphs/nodes_robots.txt'
    CalculateExpandedNodes(root_path, methods, method_names, seeds, output_file=output_file)


def Robot_ExpandedNodes_H2Full():
    seeds = range(35)

    methods = ['anytime_h2_full_2121', 'anytime_h2', 'anytime_h4']
    method_names = ['Anytime H = 2 Full', 'Anytime H = 2', 'Anytime H = 4']

    root_path = '../../robot_tests/tests1_16_ok/'
    output_file = '../../result_graphs/nodes_robot_h2_full.txt'
    CalculateExpandedNodes(root_path, methods, method_names, seeds, output_file=output_file)


if __name__ == "__main__":
    # ExpandedNodesRoads()
    # ExpandedNodesSimulated()
    # ExpandedNodesForH1()
    # ExpandedNodesRoads_H2Full()
    # Robot_ExpandedNodesForH1()
    Robot_ExpandedNodes()
    Robot_ExpandedNodes_H2Full()

import numpy as np
import os

"""
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


"""


def New_CountExpandedNodesForMethod(method_name, seeds, tests_folder, time_steps):
    total_results = np.zeros((time_steps,))
    len_seeds = len(seeds)

    for seed in seeds:
        seed_folder = tests_folder + 'seed' + str(seed) + '/' + method_name + '/'
        results = New_CountExpandedNodesForSingleSeed(path=seed_folder, time_steps=time_steps)
        total_results = np.add(total_results, results)

    return list(total_results / len_seeds)


def New_CountExpandedNodesForSingleSeed(path, time_steps):
    results = []
    for step in range(time_steps):
        current_seed_file = path + 'step' + str(step) + '.txt'
        all_summary_lines = open(current_seed_file).readlines()
        nodes_line = all_summary_lines[-1]
        expanded_node = nodes_line.split()[3]
        results.append(int(expanded_node))
    return np.array(results)


def Simulated_ExpandedNodes():
    seeds = range(66, 102)

    method = 'anytime_h4_300'
    method_name = 'Anytime H = 4'

    root_path = '../../simulated_tests/anytime/'
    time_steps = 5

    output_file = '../../result_graphs/node_files/simulated_nodes.txt'

    output_rewards = open(output_file, 'w')

    magic = 15.0 / 8.0

    results = New_CountExpandedNodesForMethod(method_name=method, seeds=seeds,
                                              tests_folder=root_path, time_steps=time_steps)
    total = results[0] * magic + results[1] * magic + sum(results[2:])
    print method_name, total
    output_rewards.write(method_name + ' ' + str(total) + '\n')

    output_rewards.close()


# Roads

def Road_ExpandedNodes():
    seeds = range(35)
    root_path = '../../road_tests/tests1/'
    time_steps = 4

    methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'new_anytime_h4_300']
    methods = ['anytime_h1', 'anytime_h2', 'anytime_h3']
    method_names = ['Anytime H = 1', 'Anytime H = 2', 'Anytime H = 3', 'Anytime H = 4 300']
    method_names = ['Anytime H = 1', 'Anytime H = 2', 'Anytime H = 3']

    output_file = '../../result_graphs/node_files/road_nodes.txt'
    output_rewards = open(output_file, 'w')

    for i, method in enumerate(methods):
        magic = 15.0 / 8 if method == 'new_anytime_h4_300' else 1.0
        results = New_CountExpandedNodesForMethod(method_name=method, seeds=seeds,
                                                  tests_folder=root_path, time_steps=time_steps)
        total = results[0] * magic + sum(results[1:])
        print method_names[i], total
        output_rewards.write(method_names[i] + ' ' + str(total) + '\n')

    output_rewards.close()


def Road_ExpandedNodes_H2Full():
    seeds = range(35)

    method = 'anytime_h2_full_2'
    method_name = 'Anytime H = 2 Full'

    root_path = '../../road_tests/21testsfull/'
    time_steps = 4

    output_file = '../../result_graphs/node_files/road_nodes.txt'
    if os.path.exists(output_file):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(output_file, append_write)

    results = New_CountExpandedNodesForMethod(method_name=method, seeds=seeds,
                                              tests_folder=root_path, time_steps=time_steps)
    total = sum(results)
    print method_name, total
    output_rewards.write(method_name + ' ' + str(total) + '\n')

    output_rewards.close()

def Road_ExpandedNodes_H4Samples():
    seeds = range(35)
    root_path = '../../road_tests/new_h4/'
    time_steps = 4

    methods = ['anytime_h4_5', 'anytime_h4_50']
    methods = ['anytime_h4_5']
    method_names = ['Anytime H = 4 5', 'Anytime H = 4 50']

    output_file = '../../result_graphs/node_files/road_nodes.txt'
    if os.path.exists(output_file):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(output_file, append_write)

    for i, method in enumerate(methods):
        magic = 15.0 / 8.0
        # magic = 15.0 / 6 if method == 'new_anytime_h4_50' else 1.0
        results = New_CountExpandedNodesForMethod(method_name=method, seeds=seeds,
                                                  tests_folder=root_path, time_steps=time_steps)
        total = results[0] * magic + sum(results[1:])
        print method_names[i], total
        output_rewards.write(method_names[i] + ' ' + str(total) + '\n')

    output_rewards.close()


# Robot

def Robot_ExpandedNodes():
    seeds = list(set(range(35)) - set([24]))
    root_path = '../../noise_robot_tests/all_tests/'
    time_steps = 4

    methods = ['anytime_h1', 'anytime_h2', 'anytime_h3', 'new_anytime_h4_300']
    method_names = ['Anytime H = 1', 'Anytime H = 2', 'Anytime H = 3', 'Anytime H = 4 300']

    output_file = '../../result_graphs/node_files/robot_nodes.txt'
    output_rewards = open(output_file, 'w')

    for i, method in enumerate(methods):
        magic = 15.0 / 6 if method == 'new_anytime_h4_300' else 1.0
        results = New_CountExpandedNodesForMethod(method_name=method, seeds=seeds,
                                                  tests_folder=root_path, time_steps=time_steps)
        total = results[0] * magic + sum(results[1:])
        print method_names[i], total
        output_rewards.write(method_names[i] + ' ' + str(total) + '\n')

    output_rewards.close()


def Robot_ExpandedNodes_H2Full():
    seeds = range(35)

    method = 'anytime_h2_full'
    method_name = 'Anytime H = 2 Full'

    root_path = '../../noise_robot_tests/21_full/'
    time_steps = 4

    output_file = '../../result_graphs/node_files/robot_nodes.txt'
    if os.path.exists(output_file):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(output_file, append_write)

    results = New_CountExpandedNodesForMethod(method_name=method, seeds=seeds,
                                              tests_folder=root_path, time_steps=time_steps)
    total = sum(results)
    print method_name, total
    output_rewards.write(method_name + ' ' + str(total) + '\n')

    output_rewards.close()


def Robot_ExpandedNodes_H4Samples():
    seeds = list(set(range(35)) - set([17]))
    root_path = '../../noise_robot_tests/h4_tests/'
    time_steps = 4

    methods = ['new_anytime_h4_5', 'new_anytime_h4_50']
    method_names = ['Anytime H = 4 5', 'Anytime H = 4 50']

    output_file = '../../result_graphs/node_files/robot_nodes.txt'
    if os.path.exists(output_file):
        append_write = 'a'
    else:
        append_write = 'w'

    output_rewards = open(output_file, append_write)

    for i, method in enumerate(methods):
        magic = 15.0 / 6
        # magic = 15.0 / 6 if method == 'new_anytime_h4_50' else 1.0
        results = New_CountExpandedNodesForMethod(method_name=method, seeds=seeds,
                                                  tests_folder=root_path, time_steps=time_steps)
        total = results[0] * magic + sum(results[1:])
        print method_names[i], total
        output_rewards.write(method_names[i] + ' ' + str(total) + '\n')

    output_rewards.close()


if __name__ == "__main__":

    # Simulated_ExpandedNodes()
    """
    Robot_ExpandedNodes()
    Robot_ExpandedNodes_H2Full()
    Robot_ExpandedNodes_H4Samples()
    """
    Road_ExpandedNodes()
    Road_ExpandedNodes_H2Full()
    Road_ExpandedNodes_H4Samples()
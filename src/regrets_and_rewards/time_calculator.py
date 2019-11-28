import numpy as np

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
        # expanded_node = nodes_line.split()[-1]
        expanded_node = nodes_line.split()[3]
        # print expanded_node, current_seed_file
        results.append(int(expanded_node))
    return np.array(results)


if __name__ == "__main__":
    batch_size = 5
    num_samples = 300

    root_path = '../../tests/robot_iter_h2_b5_s300/'
    seeds = range(0, 30)
    rewards = GetSimulatedTotalRewards(root_path=root_path,
                                       seeds=seeds,
                                       filename='robot_i_total_rewards.eps',
                                       )
    print
    regrets = GetSimulatedTotalRegrets(root_path=root_path,
                                       seeds=seeds,
                                       filename='robot_i_simple_regrets.eps',
                                       )
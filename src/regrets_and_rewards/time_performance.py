from NewSimulatedAverager import GetSimulatedTotalRewards, GetSimulatedTotalRegrets
from time_calculator import get_simulated_times


def simulated():
    batch_size = 4

    root_path = '../../tests/simulated_h3_b4/'
    root_path = '../../tests/simulated_h4/'

    seeds = range(66, 101)
    rewards = GetSimulatedTotalRewards(root_path=root_path,
                                       seeds=seeds,
                                       filename='sim_i_total_rewards.eps',
                                       )
    print
    regrets = GetSimulatedTotalRegrets(root_path=root_path,
                                       seeds=seeds,
                                       filename='sim_i_simple_regrets.eps',
                                       )

    out = get_simulated_times()
    for k in out.keys():
        for i in range(len(rewards)):
            if rewards[i][0] == k:
                out[k].append(rewards[i][1][-1])
        for i in range(len(regrets)):
            if rewards[i][0] == k:
                out[k].append(regrets[i][1][-1])

    with open(root_path + 'time_perf_data.txt', 'w') as f:
        for k, v in out.items():
            print k, v
            f.write('{},{},{},{}\n'.format(k, v[0] ,v[1], v[2]))



if __name__ == "__main__":
    simulated()
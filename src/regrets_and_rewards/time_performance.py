from NewSimulatedAverager import GetSimulatedTotalRewards, GetSimulatedTotalRegrets
from time_calculator import get_simulated_times


def simulated():
    batch_size = 4

    root_path = '../../tests/simulated_h3_b4/'
    root_path = '../../tests/simulated_h4_1/'

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

    iters = [5, 30, 50]
    iters = [25]
    names = map(str, iters)
    method_names = [r'$\epsilon$-M-GPO  $H = 4$ $N = {}$'.format(s) for s in iters]

    out = get_simulated_times(root_path=root_path,
                                   names=names,
                                   method_names=method_names)
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
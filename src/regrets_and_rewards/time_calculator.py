
def get_all_dicts(seeds, root_folder):
    results = {}
    for seed in seeds:
        seed_result = get_dict_for_one_seed(seed, root_folder)
        results[seed] = seed_result

    return  results


def get_dict_for_one_seed(seed, root_folder):
    filename =  '{}seed{}/time.txt'.format(root_folder, seed)
    result = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        split_line = line.strip().split()
        result[split_line[0]] = float(split_line[1])

    return result


def get_and_write_results(iters, method_names, seeds, results, out_file, num_stages):
    out_dict = {}
    len_seeds = len(seeds)
    for i in range(len(iters)):
        sum = 0
        for seed in seeds:
            sum+= results[seed][iters[i]]
        av_sum = sum / (len_seeds * num_stages)
        out_file.write("{}  per stage is {} \n".format(method_names[i], av_sum))
        out_dict[method_names[i]] = [av_sum]
    return out_dict


def get_anytime_times():
    root_path = '../../tests/1_road_iter_h2_b5_s300/'
    out_file = open(root_path + 'av_times.txt', 'w')

    seeds = list(set(range(0, 36)) - set([19]))
    num_stages = 4

    results = get_all_dicts(root_folder=root_path, seeds=seeds)
    iters = [200,  700, 1500]

    get_and_write_results(iters=iters,
                          seeds=seeds,
                          results=results,
                          out_file=out_file,
                          num_stages=num_stages)


def get_simulated_times(root_path, names, method_names):

    num_stages = 5
    seeds = list(set(range(66, 101)) - set([]))


    out_file = open(root_path + 'av_times.txt', 'w')
    results = get_all_dicts(root_folder=root_path, seeds=seeds)

    return get_and_write_results(iters=names,
                          method_names=method_names,
                          seeds=seeds,
                          results=results,
                          out_file=out_file,
                          num_stages=num_stages)


if __name__ == "__main__":
    # get_anytime_times()
    # root_path = '../../tests/simulated_h3_b4/'
    # names = ['MLEH=4', 'PE', 'BUCB', 'myqEI', 'LP', 'h1']
    # method_names = [r'Nonmyopic GP-UCB $H = 4$',
    #                 'GP-UCB-PE', 'GP-BUCB',
    #                 r'$q$-EI', 'BBO-LP', 'DB-GP-UCB']


    root_path = '../../tests/simulated_h4_1/'
    iters = [20, 25, 30]
    names = map(str, iters)
    method_names = [r'$\epsilon$-M-GPO  $H = 4$ $N = {}$'.format(s) for s in iters]

    h_values = get_simulated_times(root_path=root_path,
                                   names=names,
                                   method_names=method_names)
    for k in h_values.keys():
        print k, h_values[k]

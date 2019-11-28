

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
        result[int(split_line[0])] = float(split_line[1])

    return result


def get_and_write_results(iters, seeds, results, out_file):
    len_seeds = len(seeds)
    for iter in iters:
        sum = 0
        for seed in seeds:
            sum+= results[seed][iter]
        av_sum = sum / len_seeds
        out_file.write("{} iterations av. time is {} \n".format(iter, av_sum))


def get_anytime_times():
    root_path = '../../tests/1_road_iter_h2_b5_s300/'
    out_file = open(root_path + 'av_times.txt', 'w')

    seeds = list(set(range(0, 36)) - set([19]))

    results = get_all_dicts(root_folder=root_path, seeds=seeds)
    iters = [200,  700, 1500]

    get_and_write_results(iters=iters, seeds=seeds, results=results, out_file=out_file)


def get_simulated_times():
    root_path = '../../tests/simulated_h3_b4/'
    out_file = open(root_path + 'av_times.txt', 'w')

    seeds = list(set(range(66, 101)) - set([]))

    results = get_all_dicts(root_folder=root_path, seeds=seeds)

    iters = [5, 30,100]
    get_and_write_results(iters=iters, seeds=seeds, results=results, out_file=out_file)


if __name__ == "__main__":
    get_anytime_times()
    get_simulated_times()
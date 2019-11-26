import numpy as np
from scipy.stats import norm


def TupleToLine(tuple_location, dim_1, dim_2):
    float_line = tuple_location[0] * dim_2 + tuple_location[1] + 1
    return int(float_line)


def LineToTuple(line_location, dim_1, dim_2):
    return float((line_location - 1) / dim_2), float((line_location - 1) % dim_2)


# arguments are lists
def GenerateGridPairs(first_range, second_range):
    g = np.meshgrid(first_range, second_range)
    pairs = np.append(g[0].reshape(-1, 1), g[1].reshape(-1, 1), axis=1)
    return pairs


# converts ndarray to tuple
# can't pass ndarray as a key for dict
def ToTuple(arr):
    return tuple(map(tuple, arr))


# generates a set of reachable locations for simulted agent
def generate_set_of_reachable_locations(start, b_size, gap):
    steps = 20 / b_size
    a = []
    s_x, s_y = start
    for st in range(steps):
        a = a + [(round(s_x + i * gap, 2), s_y + gap * st * b_size) for i in
                 range(-20 + st * b_size, 21 - st * b_size)]
        a = a + [(s_x + gap * st * b_size, round(s_y + i * gap, 2)) for i in
                 range(-20 + st * b_size, 21 - st * b_size)]
        a = a + [(s_x - gap * st * b_size, round(s_y + i * gap, 2)) for i in
                 range(-20 + st * b_size, 21 - st * b_size)]
        a = a + [(round(s_x + i * gap, 2), s_y - gap * st * b_size) for i in
                 range(-20 + st * b_size, 21 - st * b_size)]

    return list(set(a))


def EI_Acquizition_Function(mu, sigma, best_observation):
    Z = (mu - best_observation) / sigma
    expectedImprov = (mu - best_observation) * norm.cdf(x=Z, loc=0, scale=1.0) \
                     + sigma * norm.pdf(x=Z, loc=0, scale=1.0)
    return expectedImprov


def DynamicHorizon(t, H_max, t_max):
    """
    :param t: current timestep
    :param H_max: maximum allowed horizon
    :param t_max: maximum number of timesteps
    :return:
    """
    return min(t_max - t, H_max)


def wrap_with_bucks(number, var):
    return '$%6.4f \pm %6.4f$' % (number, var)


def process_beta_name(method_name):
    split_name = method_name.split()
    assert len(split_name) == 3
    return "$\\beta=" + split_name[2] + "$"


# get rewards and regrets with variances in latex format for pasting into the table
def get_rewards_regrets_latex(rewards, regrets, process_beta=False):
    for i in range(len(rewards)):
        reward_i = rewards[i]
        regret_i = regrets[i]
        assert reward_i[0] == regret_i[0]
        method_name = process_beta_name(reward_i[0]) if process_beta else reward_i[0]
        print method_name, '&', wrap_with_bucks(reward_i[1][-1], reward_i[2][-1]), '&', \
            wrap_with_bucks(regret_i[1][-1], regret_i[2][-1]), '\\\\'


# def branin_transform(measurement):
#         val_mean = -55.291140616
#         val_std = 53.5417452487
#         return measurement* val_std + val_mean
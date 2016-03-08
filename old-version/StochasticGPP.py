import ExperimentSetup as esetup
import math
from MaximumLikelihoodGPP import CalculateMaxLikelihoodGPP
from utils import CalculateConvolution, CalculateL, GenerateStochasticSamples, GetMeasurement
from GP_calculation import Calculate_GP_Posterior

__author__ = 'Dmitrii'


def _ComputeV(t, d_t, Lambda, p):
    if t == esetup.H:
        return [0, None]
    # d_t is a list of two lists s and z
    s_t = d_t[0][-1]
    values_from_next_stage = []
    # todo
    actions_list = esetup.A(s_t)
    p_corrected = math.pow(p, (1 / float(len(actions_list))))

    for s in actions_list:
        # s is s_(t+1)
        next_value = _ComputeQ(t, s, d_t, Lambda, p_corrected)
        values_from_next_stage.append([next_value, s])
    return max(values_from_next_stage, key=lambda x: x[0])


def _ComputeQ(t, s_next, d_t, Lambda, p):
    # s_next is s_(t+1)
    nu, sigma = Calculate_GP_Posterior(s_next, d_t)
    S_next = d_t[0] + [s_next]

    r = CalculateConvolution(esetup.R_2, nu, sigma) + esetup.R_3(S_next)
    n = 2 * math.log((1 - math.sqrt(p)) / 2) * math.pow((esetup.L_1 + CalculateL(t + 1, S_next)) * sigma, 2) / ( - math
                                                                                                                 .pow(
        Lambda, 2))
    # z^i
    samples = GenerateStochasticSamples(n, nu, sigma)
    p_corrected = math.pow(p, 1 / float(2 * n))
    sample_values = map(lambda z: _ComputeV(t + 1, [S_next, d_t[1] + [z]], Lambda, p_corrected), samples)
    r_samples = map(lambda z: esetup.R_1(z), samples)
    answer_sum = sum(sample_values) + sum(r_samples)
    return r + answer_sum / n


def StochasticEpsGPP(d_0, eps):
    # preprocess
    beta = eps / esetup.H
    eps_s = beta / 4

    # for s
    # check that d_0 is just single value
    s_list = d_0[0]
    z_list = d_0[1]
    for T in range(esetup.H):
        # s_list[-1] is s_t
        actions_list = esetup.A(s_list[-1])
        Q_d_full = map(lambda s: CalculateMaxLikelihoodGPP(s, [s_list, z_list]), actions_list)
        errors_d = [v[1] for v in Q_d_full]
        Q_d = [v[0] for v in Q_d_full]
        # eps_d
        eps_d = max(errors_d)
        delta = beta / (8 * eps_d)
        Lambda = eps_s / (esetup.H - T)
        Q_s = map(lambda s: _ComputeQ(T, s, [s_list, z_list], Lambda, 1 - delta), actions_list)
        Q = []

        for i in range(len(actions_list)):
            if abs(Q_d[i] - Q_s[i]) <= eps_s + eps_d:
                Q.append(Q_s[i])
            else:
                Q.append(Q_d[i])
        best_action_index = Q.index(max(Q))
        best_action = actions_list[best_action_index]
        best_action_value = GetMeasurement(best_action)
        # add best values to data for GP training
        s_list.append(best_action)
        z_list.append(best_action_value)
        # todo check what is TakeAction after determinitic version works

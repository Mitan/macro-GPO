import ExperimentSetup as esetup
from utils import CalculateConvolution, GenerateDeterministicSamplesAndWeights
from GP_calculation import Calculate_GP_Posterior


def _ComputeV(t, d_t, Lambda, numberOfSamples):
    if t == esetup.H:
        return [0, None]
    # d_t is a list of two lists s and z
    s_t = d_t[0][-1]
    values_from_next_stage = []
    # todo
    actions_list = esetup.A(s_t)

    for s in actions_list:
        # s is s_(t+1)
        next_value = _ComputeQ(t, s, d_t, Lambda, numberOfSamples)
        values_from_next_stage.append([next_value, s])
    # return max by Q-value
    return max(values_from_next_stage, key=lambda x: x[0])


def _ComputeQ(t, s_next, d_t, Lambda, numberOfSamples):
    # s_next is s_(t+1)
    nu, sigma = Calculate_GP_Posterior(s_next, d_t)
    S_next = d_t[0] + [s_next]

    r = CalculateConvolution(esetup.R_2, nu, sigma) + esetup.R_3(S_next)

    # z^i
    samples, weights = GenerateDeterministicSamplesAndWeights(numberOfSamples, nu, sigma)
    sample_values = map(lambda z: _ComputeV(t + 1, [S_next, d_t[1] + [z]], Lambda, numberOfSamples), samples)
    r_samples = map(lambda z: esetup.R_1(z), samples)

    answer_sum = [(x + y) * w for x in sample_values for y in r_samples for w in weights]
    return r + sum(answer_sum)


def DeterministicEpsGPP(d_0, eps, numberOfSamples):
    # preprocess

    Lambda = eps / (esetup.H * (esetup.H + 1))
    value, action = _ComputeV(0, d_0, Lambda, numberOfSamples)
    return action

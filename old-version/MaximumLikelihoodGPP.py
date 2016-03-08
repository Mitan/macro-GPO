import ExperimentSetup as esetup
import math
from GP_calculation import Calculate_GP_Posterior
from utils import CalculateConvolution, CalculateL


def _ComputeV(t, d_t):
    if t == esetup.H:
        return [0, None]
    # d_t is a list of two lists s and z
    s_t = d_t[0][-1]
    values_from_next_stage = []
    # todo
    actions_list = esetup.A(s_t)

    for s in actions_list:
        # s is s_(t+1)
        next_value = CalculateMaxLikelihoodGPP(t, s, d_t)
        values_from_next_stage.append([next_value, s])
    # return max by Q-value
    return max(values_from_next_stage, key=lambda x: x[0])


def _CalculateError(t, sigma, S_next):
    # formula from page 30
    # tau == 0
    # S_next is S_(t+1)
    return math.sqrt(2 / math.pi) * sigma * (esetup.L_1 + CalculateL(t + 1, S_next))


def CalculateMaxLikelihoodGPP(t, s_next, d_t):
    # s_next is s_(t+1)
    nu, sigma = Calculate_GP_Posterior(s_next, d_t)
    # S_next is S_(t+1)
    S_next = d_t[0] + [s_next]

    r = CalculateConvolution(esetup.R_2, nu, sigma) + esetup.R_3(S_next)
    # return value from jsut one sample - posterior mean
    answer = r + esetup.R_1(nu) + _ComputeV(t + 1, [S_next, d_t[1] + [nu]])
    error = _CalculateError(t, sigma, S_next)
    return [answer, error]
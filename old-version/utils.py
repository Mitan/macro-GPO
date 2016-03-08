import math

from numpy import random, inf
from scipy.integrate import quad

import ExperimentSetup as esetup
from GP_calculation import CalculateAlpha, Calculate_GP_Posterior


def _calculateKernelForConvolution(f, nu, sigma, x):
    c = 1.0 / (sigma * math.sqrt(2 * math.pi))
    return c * f(*x) * math.exp(- math.pow(nu - x[0], 2) / (2 * math.pow(sigma, 2)))


# g in notation of alg
def CalculateConvolution(f, nu, sigma):
    return quad(lambda x: _calculateKernelForConvolution(f, nu, sigma, [x]), -inf, inf)


# todo think how to optimize
def CalculateL(t, S_t):
    if t == esetup.H:
        return 0
    # S_(t+1)
    s_t = S_t[-1]
    actions = esetup.A(s_t)
    maximums = []
    for x in actions:
        alpha = CalculateAlpha(x, S_t)
        current_value = alpha * (esetup.L_1 + esetup.L_2) + math.sqrt(1 + alpha * alpha) * CalculateL(t + 1, S_t + [x])
        maximums.append(current_value)
    return max(maximums)


# get samples from normal distribution
def GenerateStochasticSamples(n, nu, sigma):
    return random.normal(nu, sigma, n)


def _normalPDF(x):
    return math.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)


def GenerateDeterministicSamplesAndWeights(n, nu, sigma):
    # user defined hardcoded
    tau = 3
    samples = []
    weights = []
    z0 = nu - tau * sigma
    zLast = nu + sigma * tau
    samples.append(z0)
    weights.append(_normalPDF(-tau))
    for i in range(1, n - 1):
        z_i = z0 + (i - 0.5) * (zLast - z0) / (n - 2)
        w_i = _normalPDF(2 * i * tau / float(n - 2) - tau) - _normalPDF(2 * (i - 1) * tau / float(n - 2) - tau)
        samples.append(z_i)
        weights.append(w_i)
    samples.append(zLast)
    weights.append(_normalPDF(-tau))
    return [samples, weights]


# return z
def GetMeasurement(s_next, d_t):
    nu, sigma = Calculate_GP_Posterior(s_next, d_t)
    return nu




# test
"""
def t(x):
    return 1

print CalculateConvolution(t, 0.0, 1.0)
"""
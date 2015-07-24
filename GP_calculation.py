import math

import numpy as np
from scipy import linalg

import ExperimentSetup as esetup


__author__ = 'Dmitrii'

"""test input """
noise = np.random.normal(0.0, 0.3, 6)
inputs = [-1.5, -1.0, -0.75, -0.4, -0.25, 0.0]
y_clean = [-1.6, -1.05, -0.3, 0.25, 0.5, 0.75]
y_noise = [x + y for (x, y) in zip(noise, y_clean)]
x_test = 0.2


def _deltaFunction(x, y):
    return 1 if x == y else 0


# in 2-d case x and y a 2-d vectors
# s.T *  M^-2 * s = s_1^2 / l_1^2  + s_2^2 / l_2^2
def _k(x, y):
    s_1 = x[0] - y[0]
    s_2 = x[1] - y[1]
    product = s_1 ** 2 / float(esetup.l_1 ** 2) + s_2 ** 2 / float(esetup.l_2 ** 2)
    return math.pow(esetup.sigma_y, 2) * math.exp(- product) + math.pow(esetup.sigma_n, 2) * _deltaFunction(x, y)


def _getCovarianceMatrix(in_values):
    length = len(in_values)
    covariance_list = [[_k(in_values[i], in_values[j]) for j in range(length)] for i in range(length)]
    return np.matrix(covariance_list)


def Calculate_GP_Posterior(s_next, d_t):
    inputs = d_t[0]
    values = d_t[1]
    # K_s is K_*
    K = _getCovarianceMatrix(inputs)
    K_s_s = K[0, 0]
    K_s = np.asarray([_k(s_next, v) for v in inputs])

    #solve linear system with cholesky, from Rassmussen book p 37
    L = np.linalg.cholesky(K)
    temp = linalg.solve_triangular(L, values, lower=True)
    alpha = linalg.solve_triangular(L.T, temp)
    v = linalg.solve_triangular(L, K_s.T, lower=True)
    nu = np.dot(K_s, alpha)
    sigma = K_s_s - np.dot(v.T, v)
    return [nu, sigma]


# alpha from Lemma 1 in Lipshitz functions
# for alpha we need only inputs
# use the trick - we need to calculate row * A^-1. Use a transpose to get A.T ^-1 * column
# since covariance matrix is symmetric, A.T = A. So we calculate A^-1 * column
def CalculateAlpha(s_next, S_t):
    K = _getCovarianceMatrix(S_t)
    L = np.linalg.cholesky(K)
    K_s = np.asarray([_k(s_next, v) for v in inputs]).T
    temp = linalg.solve_triangular(L, K_s, lower=True)
    alpha_vector = linalg.solve_triangular(L.T, temp)
    return np.linalg.norm(alpha_vector)




    #print Calculate_GP_Posterior(x_test, [inputs, y_noise])
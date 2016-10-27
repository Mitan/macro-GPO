import GPy
import numpy as np
import math
import random

import scipy

noise_variance = random.random()  # Random float x, 0.0 <= x < 1.0
l = random.uniform(1, 10)
signal_variance = random.uniform(2, 5)


def SquareExpKernel(x, y):
    eps_tolerance = 10**-6
    ind = ((x - y) / l) ** 2
    kronecker = 1 if np.linalg.norm(x - y) < eps_tolerance else 0
    return signal_variance * math.exp(- 0.5 * ind) + kronecker * noise_variance


def CovarianceMesh(col_array, row_array):
    """
    @param col_array, row - array of shape (number of dimensions * number of data points)
    @return covariance matrix between physical states presented by col and row
    """
    columns = col_array.shape[0]
    rows = row_array.shape[0]
    covMat = np.zeros((columns, rows), float)
    for y in xrange(columns):
        for x in xrange(rows):
            covMat[y, x] = SquareExpKernel(row_array[x, :], col_array[y, :])
            # covMat[y, x] = row_array[x, :][0] + col_array[y, :][0]
    return covMat


def MyPredict(new_loc):
    K = CovarianceMesh(locations, locations)
    k_star = CovarianceMesh(locations, new_loc)
    # print k_star.shape
    number_of_points = locations.shape[0]
    # print K
    # noise_matrix = noise_variance * np.identity(number_of_points)
    # assert noise_matrix.shape == K.shape
    # K_noise = K + noise_matrix
    # L is lower triangular, L *  L.T = K_noise
    L = np.linalg.cholesky(K)

    temp = scipy.linalg.solve_triangular(L, Y, lower=True)
    alpha = scipy.linalg.solve_triangular(L.T, temp, lower=False)
    mu = np.dot(k_star.T, alpha)

    v = scipy.linalg.solve_triangular(L, k_star, lower=True)
    var = signal_variance + noise_variance - np.dot(v.T, v)
    return mu, var


def GPyPredict(new_loc):
    kernel = GPy.kern.RBF(input_dim=1, variance=signal_variance, lengthscale=l)
    m = GPy.models.GPRegression(locations, Y, kernel, noise_var=noise_variance, normalizer=False)
    return m.predict(new_loc)


if __name__ == "__main__":
    for k in range(50):
        locations = np.random.uniform(-3., 3., (20, 1))
        Y = np.sin(locations) + np.random.randn(20, 1) * 0.05
        Y = Y - np.mean(Y)
        new_location = np.random.uniform(-3., 3., (1, 1))

        mu, var = MyPredict(new_location)
        mu_gpy, var_gpy = GPyPredict(new_location)

        eps_tolerance = 10 ** -6
        assert np.linalg.norm(mu- mu_gpy) < eps_tolerance
        assert np.linalg.norm(var- var_gpy) < eps_tolerance

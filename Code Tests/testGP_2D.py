import GPy
import numpy as np
import math
import random

import scipy

noise_variance = random.random()  # Random float x, 0.0 <= x < 1.0
l_1 = random.uniform(1, 10)
l_2 = random.uniform(1, 10)
signal_variance = random.uniform(2, 5)


# 2D
def SquareExpKernel(x, y):
    eps_tolerance = 10 ** -6
    kronecker = 1 if np.linalg.norm(x - y) < eps_tolerance else 0
    ind = ((x[0] - y[0]) / l_1) ** 2 + ((x[1] - y[1]) / l_2) ** 2
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
    K_current = CovarianceMesh(new_loc, new_loc)
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
    v_prodcut = np.dot(v.T, v)
    assert v_prodcut.shape == K_current.shape
    var = K_current - v_prodcut
    return mu, var


def GPyPredict(new_loc):
    kernel = GPy.kern.RBF(input_dim=2, variance=signal_variance, lengthscale=[l_1, l_2], ARD= True)
    m = GPy.models.GPRegression(locations, Y, kernel, noise_var=noise_variance, normalizer=False)
    return m.predict(new_loc, full_cov=True)


if __name__ == "__main__":
    for k in range(100):

        num_points = 20
        locations = np.random.uniform(-3., 3., (num_points, 2))
        # change into 1D
        Y = np.zeros((num_points,1))
        for i in range(num_points):
            current_point = locations[i, :]
            Y[i, 0] = -0.2 * current_point[0] + current_point[1]**2

        Y = Y - np.mean(Y)
        new_location = np.random.uniform(-3., 3., (10, 2))

        mu, var = MyPredict(new_location)
        mu_gpy, var_gpy = GPyPredict(new_location)

        eps_tolerance = 10 ** -6

        assert mu.shape == mu_gpy.shape
        assert np.linalg.norm(mu - mu_gpy) < eps_tolerance

        assert var.shape == var_gpy.shape
        assert np.linalg.norm(var- var_gpy) < eps_tolerance, "difference is %r" % np.linalg.norm(var- var_gpy)
import random

import GPy
import numpy as np

from src.GaussianProcess import SquareExponential, GaussianProcess


def MyPredict(new_loc):
    covariance_function = SquareExponential(length_scale, signal_variance, noise_variance)
    gp = GaussianProcess(covariance_function=covariance_function, mean_function=0.0)
    cholesky = gp.Cholesky(locations)
    weights = gp.GPWeights(locations=locations, current_location=new_loc, cholesky=cholesky)
    var = gp.GPVariance(locations=locations, current_location=new_loc, cholesky=cholesky)
    mu = gp.GPMean(measurements=Y, weights=weights)
    return mu, var


def GPyPredict(new_loc):
    kernel = GPy.kern.RBF(input_dim=point_dimension, variance=signal_variance, lengthscale=length_scale, ARD= True)
    m = GPy.models.GPRegression(locations, Y, kernel, noise_var=noise_variance, normalizer=False)
    return m.predict(new_loc, full_cov=True)


if __name__ == "__main__":
    for k in range(100):

        # init dimensions
        point_dimension = random.randint(1, 9)
        num_points = 20

        # init hypers
        noise_variance = random.random()  # Random float x, 0.0 <= x < 1.0
        signal_variance = random.uniform(2, 5)
        length_scale = [random.uniform(5., 10.0) for i in range(point_dimension)]

        # init setting
        locations = np.random.uniform(-10., 15., (num_points, point_dimension))

        Y = np.zeros((num_points,1))
        for i in range(num_points):
            current_point = locations[i, :]
            Y[i, 0] = -0.2 * current_point[0] + current_point[0]**2 + np.sum(current_point)

        Y = Y - np.mean(Y)
        new_location = np.random.uniform(-3., 3., (2, point_dimension))

        # predict
        mu, var = MyPredict(new_location)
        mu_gpy, var_gpy = GPyPredict(new_location)

        eps_tolerance = 10 ** -5

        # compare
        assert mu.shape == mu_gpy.shape
        assert np.linalg.norm(mu - mu_gpy) < eps_tolerance, "difference is %r" % np.linalg.norm(mu - mu_gpy)

        assert var.shape == var_gpy.shape
        assert np.linalg.norm(var - var_gpy) < eps_tolerance, "difference is %r" % np.linalg.norm(var - var_gpy)


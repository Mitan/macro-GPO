import GPy
import math
import numpy as np
from Old_GaussianProcess import SquareExponential, GaussianProcess

__author__ = 'a0134673'

lengthscale = [5.0, 2.0]
signal = 0.3
noise = 0.03

def f(x):
    return math.sqrt(x[0]**2 + x[1]**2)


X = np.arange(-2.0, 2.0, step= 0.2)
Y = np.arange(-2.0, 2.0, step= 0.2)
grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
values = np.apply_along_axis(f, 1, grid)
values = values.reshape((values.shape[0], 1))


def GPY_predict(test_point):
    kernel = GPy.kern.RBF(input_dim=2,variance= signal,lengthscale=lengthscale, ARD= True)
    m = GPy.models.GPRegression(grid,values,kernel, noise_var=noise)
    mu_gp, sigma_gp =  m.predict(test_point, full_cov=True)
    return mu_gp,sigma_gp


def ChunKai_predict(test_point):
    covariance_function = SquareExponential(np.array(lengthscale), signal)
    gp = GaussianProcess(covariance_function, noise, mean_function=0.0)
    chol = gp.GPCholTraining(grid)
    Sigma_chun = gp.GPBatchVariance(grid, test_point, chol)
    weights = gp.GPBatchWeights(grid, test_point, chol)
    mu_chun = gp.GPBatchMean(values, weights)
    return mu_chun, Sigma_chun

test_point = np.asarray([[1.5, 2.3], [3.4, 2.7]])

for i in range(100):
    test_point =  np.random.rand(2,2)
    mu_chun, sigma_chun = ChunKai_predict(test_point)
    mu_gp, sigma_gp = GPY_predict(test_point)

    mu_diff = np.linalg.norm(mu_chun - mu_gp)
    sigma_diff = np.linalg.norm(sigma_chun - sigma_gp)
    if mu_diff > 0.001:
        print mu_diff
    if sigma_diff  > 0.001:
        print sigma_diff

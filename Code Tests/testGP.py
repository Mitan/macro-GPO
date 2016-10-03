import GPy
import math
import numpy as np
#from GaussianProcess import SquareExponential, GaussianProcess
#from GaussianProcess import SquareExponential
from src.GaussianProcess import GaussianProcess, SquareExponential

__author__ = 'a0134673'


def f(x):
    return math.sqrt(2 * x[0]**2 + 23 * x[1]**2)
"""
X = np.arange(-2.0, 2.0, step= 0.2)
Y = np.arange(-2.0, 2.0, step= 0.2)
grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
values = np.apply_along_axis(f, 1, grid)
values = values.reshape((values.shape[0], 1))


X = np.random.uniform(-3.,3.,(20,2))
Y = np.sin(X) + np.random.randn(20,2)*0.1

kernel = GPy.kern.RBF(input_dim=2,variance= signal,lengthscale=lengthscale, ARD= True)
m = GPy.models.GPRegression(X,Y,kernel, noise_var=noise)
m.optimize_restarts(num_restarts = 10)
"""

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


def MyPredictiSinglePoint(test_point):
    covariance_function = SquareExponential(np.array(lengthscale), signal)
    gp = GaussianProcess(covariance_function, noise, mean_function=0.0)
    mean = gp.GPMean(grid, values, test_point)
    var = gp.GPVariance2(grid, test_point)
    return mean, var
#test_point = np.asarray([[1.5, 2.3], [3.4, 2.7]])


for k in range(10):
    dim = 3
    num_points = 10
    X = np.random.uniform(-3., 3., (num_points, dim))
    X = X - np.mean(X, axis=0)

    Y = np.apply_along_axis(f, 1, X).reshape((num_points,1)) + np.random.randn(num_points, 1) * 0.2
    #Y = np.sin(X) + np.random.randn(10, dim) * 0.2
    #print Y
    length_scale = [1.0] * dim

    diff_chun = 0
    diff_gpy = 0

    for i in range(100):
        test_point = np.random.rand(1, dim)
        test_value = np.sin(test_point) + np.random.rand(1, dim) * 0.2
        # kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=[1.0, 1.0], ARD=True)
        kernel = GPy.kern.RBF(input_dim=dim, variance=1.0, lengthscale=length_scale, ARD=True)

        m = GPy.models.GPRegression(X, Y, kernel, noise_var=0.01)
        # print m
        m.optimize(messages=False)
        # print m
        mu_gpy = m.predict(test_point)[0]
        # params
        signal =  m.param_array[0]
        length_scale = m.param_array[1:dim+1]
        noise = m.param_array[dim+1]
        # print length_scale
        """
        covariance_function = SquareExponential(np.array(length_scale), signal)
        gp = GaussianProcess(covariance_function, noise, mean_function=0.0)
        chol = gp.GPCholTraining(X)
        weights = gp.GPBatchWeights(X, test_point, chol)
        mu_chun = gp.GPBatchMean(Y, weights)
        """
        covariance_function = SquareExponential(np.array(length_scale), signal)
        gp = GaussianProcess(covariance_function, noise, mean_function=0.0)
        mu_chun = gp.GPMean(X, Y, test_point)

        diff_chun += np.linalg.norm(mu_chun - test_value)**2
        diff_gpy += np.linalg.norm(mu_gpy - test_value)**2


    #   print "Chun Kai {}".format( math.sqrt(diff_chun))
    #   print "GPY {}" .format(math.sqrt(diff_gpy))
    best = "Chun" if diff_chun < diff_gpy else "GPy"
    print best
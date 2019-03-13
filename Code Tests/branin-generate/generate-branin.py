import datetime

import GPy
import math
import numpy as np
import scipy


a = 1.0
b = 5.1 / (4 * math.pi ** 2)
c = 5 / math.pi
r = 6.0
s = 10.0
t = 1 / (8 * math.pi)


def branin_function(x):
    return a* (x[1] - b*x[0]**2 + c* x[0] - r)**2 + s*(1-t)* math.cos(x[0]) + s


def optimize_model(X, Y):
    num_points = X.shape[0]
    assert num_points == 400
    # sample = np.random.randint(low=0, high=num_points - 1, size=500)
    # Y = Y[sample, :]
    outfile = open("./hypers/hypers1.txt", 'w')

    outfile.write("aa")

    # X = X[sample, :]


    # print X.shape, Y.shape

    k = GPy.kern.RBF(input_dim=2, ARD=True)
    m = GPy.models.GPRegression(X, Y, k)
    m.constrain_bounded(1e-2, 1e4)



    outfile.write(m.param_array)
    # m.likelihood.variance.fix(1.0)
    # m.constrain_bounded(lower=1.0, upper=1000.)
    #m.kern.variance.fix(1.0)
    # print m
    m.randomize()
    # m.optimize()
    m.optimize(messages=False)
    m.optimize_restarts(num_restarts=20)
    print m.param_array
    outfile.write(m.param_array)
    print m


def check_branin_correctness():
    x_1 = (-math.pi, 12.275)

    x_2 = (9.42478, 2.475)
    print branin_function(x_1)
    print branin_function(x_2)


def get_brainin_points(num_samples, grid_domain):

    # Number of dimensions of the multivariate gaussian is equal to the number of grid points
    ndims = len(num_samples)
    grid_res = [float(grid_domain[x][1] - grid_domain[x][0]) / float(num_samples[x]) for x in xrange(ndims)]
    npoints = reduce(lambda a, b: a * b, num_samples)

    # Mean function is assumed to be zero

    # List of points
    grid1dim = [slice(grid_domain[x][0], grid_domain[x][1], grid_res[x]) for x in xrange(ndims)]
    grids = np.mgrid[grid1dim]
    points = grids.reshape(ndims, -1).T
    # vals = - np.atleast_2d(vector_branin(points)).T
    vals = - vector_branin(points)
    # print vals.shape
    return points, vals


if __name__ == "__main__":

    vector_branin = np.vectorize(branin_function, signature='(m)->()')

    num_samples = (20, 20)

    grid_domain = ((-5.0, 10.0), (0, 15))
    points, vals = get_brainin_points(num_samples=num_samples,
                                      grid_domain=grid_domain)
    # dataset = np.concatenate((points, vals), axis=1)
    # np.savetxt(X=dataset, fname='./branin_1600points_inverse_sign.txt', fmt='%10.6f')
    vals_mean = np.mean(vals)
    vals_std = np.std(vals)
    print scipy.stats.skew(vals)


    vals = (vals - vals_mean) / vals_std
    points_mean = np.mean(points, axis=0)
    points_std = np.std(points, axis=0)

    print vals_mean,vals_std
    # print points_std
    # print max(vals)

    vals = np.atleast_2d(vals).T
    # points = np.divide(points - points_mean, points_std)
    dataset = np.concatenate((points, vals), axis=1)

    # np.savetxt(X=dataset, fname='./branin_1600points_inverse_sign_normalised.csv', delimiter=',',fmt='%10.6f')
    # np.savetxt(X=dataset, fname='./branin_400points_inverse_sign_normalised.txt',fmt='%10.6f')
    # optimize_model(X=points, Y=np.atleast_2d(vector_branin(points)).T)

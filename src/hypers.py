import GPy
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.random import normal
from random import choice
from math import sqrt


def TestPrediction(locs, vals):
    test_num_point = 20
    signal = 1.0
    noise = 0.1
    mu = 0.0
    l_1 = 1.0
    l_2 = 1.0

    vals = vals - mu

    number_of_points = locs.shape[0]
    array_range = np.arange(number_of_points)
    indexes = np.random.choice(array_range, test_num_point).tolist()
    train = indexes[:test_num_point / 2]
    test = indexes[test_num_point / 2:]
    locs_train = locs[train, :]
    vals_train = vals[train, :]

    kernel = GPy.kern.RBF(input_dim=2, variance=signal, lengthscale=[l_1, l_2], ARD=True)
    m = GPy.models.GPRegression(locs_train, vals_train, kernel, noise_var=noise, normalizer=False)
    error = 0
    for test_p in test:
        predict =  m.predict(locs[test_p, :], full_cov=True) + mu
        truth = vals[test_p]
        error += (predict - truth)**2

    print sqrt(error/ (test_num_point / 2))


# required to provide initial guess for hypers
def InferHypers(X, Y, noise, signal, l_1, l_2):
    mu = np.mean(Y)
    Y = Y - mu
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == 2

    kernel = GPy.kern.RBF(input_dim=2, variance=signal, lengthscale=[l_1, l_2], ARD=True)
    m = GPy.models.GPRegression(X, Y, kernel, noise_var=noise)
    # m.constrain_bounded('rbf_var',1e-3,1e5)
    # m.constrain_bounded('rbf_len',.1,200.)
    # m.constrain_fixed('noise',1e-5)

    # print m
    # print m

    # m.optimize(messages=True)
    m.optimize(messages=False)
    m.optimize_restarts(num_restarts=10)

    # lengthscales go indexes 1 and 2
    # todo note need to square the l_1 and l_2
    l_1, l_2 = m.param_array[1:3]
    # print l_1, l_2

    # todo note this is already sigma^2
    noise_variance = m.param_array[3]
    # print noise_variance

    signal_variance = m.param_array[0]
    # print signal_variance
    # print m
    # return l_1, l_2, noise_variance, signal_variance
    print m
    """
    out_file = open('hypers_learnt.txt', 'w')
    out_file.write(str(mu) + ' ' + str(l_1) + ' ' + str(l_2) + ' ' + str(noise_variance) + ' ' + str(signal_variance))
    out_file.close()
    """
    return mu, l_1, l_2, noise_variance, signal_variance
    # return m, mu


if __name__ == "__main__":
    filename = './hypers18.txt'
    data = np.genfromtxt(filename)
    # print data
    locs = data[:, :2]
    vals = data[:, 2]
    number_of_points = vals.shape[0]
    print scipy.stats.skew(vals), np.mean(vals), np.std(vals)

    normals = np.random.normal(scale=0.1, size=number_of_points)
    normals = np.abs(normals)

    print max(normals), np.mean(normals)
    assert normals.shape == vals.shape

    vals = np.add(vals, normals)
    print scipy.stats.skew(vals)

    vals = np.log(vals + 0.1)

    print scipy.stats.skew(vals)

    plt.hist(vals, bins=100)

    # plt.show()


    vals = np.atleast_2d(vals).T

    signal = (max(vals) - min(vals)) / 2
    l_1 = 2.0 / 25.0
    l_2 = 2.0 / 50.0
    noise = 0.1 * signal
    print InferHypers(X=locs, Y=vals, noise=noise, signal=signal, l_1=l_1, l_2=l_2)

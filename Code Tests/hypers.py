import GPy
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.random import normal
from random import choice
from math import sqrt, exp


def TestPrediction(locs, vals):

    test_num_point = 20
    # l_1, l_2, signal, noise
    params = np.array([-2.525728644308256,-3.218875824868201,1.827078296982567, -0.165623872912252])
    params = np.exp(params)

    mu = -4.638786853251777

    signal = params[2]
    noise = params[3]

    l_1 = params[0]
    l_2 = params[1]
    print signal, noise,  l_2

    vals = vals - mu

    number_of_points = locs.shape[0]
    array_range = np.arange(number_of_points)
    # lists
    indexes = np.random.choice(array_range, test_num_point).tolist()
    half_points = test_num_point / 2
    train = indexes[:half_points]
    test = indexes[half_points:]
    # 2d array
    locs_train = locs[train, :]
    # print locs_train
    1# d array
    vals_train = vals[train]
    # print vals_train
    vals_train = np.atleast_2d(vals_train).T

    kernel = GPy.kern.RBF(input_dim=2, variance=signal, lengthscale=[l_1, l_2], ARD=True)
    m = GPy.models.GPRegression(locs_train, vals_train, kernel, noise_var=noise, normalizer=False)
    point = np.array([[1,2]])
    print m.predict(point)
    error = 0
    for test_p in test:
        point = locs[test_p: test_p + 1, :]
        predict = m.predict(point, full_cov=False)
        predict_mean = (predict[0])[0,0] + mu
        truth = vals[test_p]
        error += (predict_mean - truth)**2

    print sqrt(error / (half_points))


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
    m.optimize(messages=True)
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
    filename = './hypers44.txt'
    data = np.genfromtxt(filename)
    # print data
    locs = data[:, :2]
    vals = data[:, 2]
    number_of_points = vals.shape[0]
    # print scipy.stats.skew(vals), np.mean(vals), np.std(vals)

    normals = np.random.normal(scale=0.1, size=number_of_points)
    normals = np.abs(normals)

    # print max(normals), np.mean(normals)
    assert normals.shape == vals.shape

    # vals = np.add(vals, normals)
    # print scipy.stats.skew(vals)

    f = 0.00001
    vals = np.log(vals + f)
    # print "f = " + str(f)

    # print scipy.stats.skew(vals)
    # print np.mean(vals), np.std(vals)
    """
    plt.hist(vals, bins=100)

    # plt.show()


    vals = np.atleast_2d(vals).T

    signal = ((max(vals) - min(vals)) / 2)[0]
    print
    print signal
    print np.mean(vals)
    l_1 = 2.0 / 25.0
    l_2 = 2.0 / 50.0
    noise = 0.1 * signal
    # print InferHypers(X=locs, Y=vals, noise=noise, signal=signal, l_1=l_1, l_2=l_2)
    """
    print
    for i in range(1):
        TestPrediction(locs, vals)
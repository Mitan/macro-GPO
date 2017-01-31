import GPy
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.random import normal

def TestPrediction():
    # choice 10
    # 10 to learn 10 to test
    pass



# required to provide initial guess for hypers
def InferHypers(X, Y, noise, signal, l_1, l_2):
    mu = np.mean(Y)
    Y = Y - mu
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] ==2

    kernel = GPy.kern.RBF(input_dim=2,variance= signal,lengthscale=[l_1, l_2], ARD= True)
    m = GPy.models.GPRegression(X,Y,kernel, noise_var=noise)
    #m.constrain_bounded('rbf_var',1e-3,1e5)
    #m.constrain_bounded('rbf_len',.1,200.)
    #m.constrain_fixed('noise',1e-5)

    #print m
    #print m

    #m.optimize(messages=True)
    m.optimize(messages=False)
    m.optimize_restarts(num_restarts = 10)


    # lengthscales go indexes 1 and 2
    #todo note need to square the l_1 and l_2
    l_1, l_2 =  m.param_array[1:3]
    #print l_1, l_2

    # todo note this is already sigma^2
    noise_variance = m.param_array[3]
    #print noise_variance

    signal_variance = m.param_array[0]
    #print signal_variance
    #print m
    #return l_1, l_2, noise_variance, signal_variance
    print m
    return mu, l_1, l_2, noise_variance,signal_variance
    # return m, mu

if __name__ == "__main__":
    filename = './hypers44.txt'
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

    vals = np.log(vals+ 0.01)


    print scipy.stats.skew(vals)


    plt.hist(vals, bins=100)

    plt.show()

    """
    vals = np.atleast_2d(vals).T

    print locs.shape, vals.shape

    signal = max(vals) - min(vals)
    noise = 0.1 * signal
    print InferHypers(locs,vals, noise, signal, 5.0, 10.0)
    """

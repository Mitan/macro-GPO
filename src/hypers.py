"""
import GPy
import numpy as np

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

    #return mu, l_1, l_2, noise_variance,signal_variance
    return m, mu

if __name__ == "__main__":
    X = np.random.uniform(-3.,3.,(400,2))
    Y = np.sin( np.sum(X, axis=0) ) + np.random.randn(400,1)*0.05
    #print InferHypers(X, Y)
"""
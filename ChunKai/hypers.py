import GPy
import numpy as np



"""
print X.shape
print Y.shape
"""
def InferHypers(X, Y):
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] ==2
    kernel = GPy.kern.RBF(input_dim=2, lengthscale=[1.0, 1.0], ARD= True)
    m = GPy.models.GPRegression(X,Y,kernel)
    #print m

    #m.optimize(messages=True)
    m.optimize(messages=False)
    m.optimize_restarts(num_restarts = 10)


    # lengthscales go indexes 1 and 2
    l_1, l_2 =  m.param_array[1:3]
    #print l_1, l_2

    noise_variance = m.param_array[3]
    #print noise_variance

    signal_variance = m.param_array[0]
    #print signal_variance
    print m
    return l_1, l_2, noise_variance, signal_variance

if __name__ == "__main__":
    X = np.random.uniform(-3.,3.,(400,2))
    Y = np.sin( np.sum(X, axis=0) ) + np.random.randn(400,1)*0.05
    print InferHypers(X, Y)

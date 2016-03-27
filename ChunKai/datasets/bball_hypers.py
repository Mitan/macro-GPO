#from ChunKai.hypers import InferHypers
import numpy as np
#import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
"""
def InferHypers(X, Y, noise, signal, l_1, l_2):

    Y = Y - np.mean(Y)
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] ==2

    kernel = GPy.kern.RBF(input_dim=2,variance= signal,lengthscale=[l_1, l_2], ARD= True)
    m = GPy.models.GPRegression(X,Y,kernel, noise_var=noise)
    print m
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
    print m
    #return l_1, l_2, noise_variance, signal_variance
    return l_1, l_2, noise_variance,signal_variance

"""



file = open("./bball.dat")
data = np.genfromtxt(file,skip_header=10)
file.close()

# restrict the field
indexes_x = [i for i in range(data.shape[0]) if data[i,0] > 4  and data[i,1] < 19 and data[i,1] > 6]
# restricted full data
data = data[indexes_x, :]


X_values = data[:, 0:2]
print X_values

#print max(data[:, 0:1])
#K_normal = data[:, 2:3]
K_log = data[:, 3:4]
#P_normal = data[:, 5:6]
P_log = data[:, 6:7]
#print K_log



print X_values.shape
#print InferHypers(X_values, K_log, 0.02, 0.057, 1.1, 2.5)

plt.plot(*zip(*X_values), marker='o', color='r', ls='')
plt.show()




x = K_log

# the histogram of the data
n, bins, patches = plt.hist(K_log, 50, normed=1, facecolor='green', alpha=0.75)


plt.axis([0, 2, 0, 10])
plt.grid(True)

#plt.show()

#from ChunKai.hypers import InferHypers
import numpy as np
import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
#from MapDatasetStorage import MapDatasetStorage
from MapDatasetStorage import MapDatasetStorage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


def InferHypers(X, Y, noise, signal, l_1, l_2):

    Y = Y - np.mean(Y)
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] ==2

    kernel = GPy.kern.RBF(input_dim=2,variance= signal,lengthscale=[l_1, l_2], ARD= True)
    m = GPy.models.GPRegression(X,Y,kernel, noise_var=noise)
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
    print m
    #return l_1, l_2, noise_variance, signal_variance
    return l_1, l_2, noise_variance,signal_variance




file = open("./bball.dat")
data = np.genfromtxt(file,skip_header=10)
file.close()
"""
X_values = data[:, 0:2]
plt.plot(*zip(*X_values), marker='o', color='r', ls='')
K_normal = data[:, 2:3]
#K_log = data[:, 3:4]
K_log = np.log(K_normal)

x = data[:, 0:1]
y = data[: 1:2]
X, Y = np.meshgrid(x,y)

K_set = MapDatasetStorage(X_values, K_log)

def getPoint(x,y):
    return K_set([x,y])

vec_point = np.vectorize(getPoint)

Z = vec_point(X,Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
"""
# restrict the field
indexes_x = [i for i in range(data.shape[0]) if data[i,0] > 4  and data[i,1] < 18 and data[i,1] > 5]
# restricted full data
data = data[indexes_x, :]

X_values = data[:, 0:2]
plt.plot(*zip(*X_values), marker='o', color='r', ls='')


#plt.show()

X_values = data[:, 0:2]
#print X_values

#print max(data[:, 0:1])
K_normal = data[:, 2:3]
#K_log = data[:, 3:4]
K_log = np.log(K_normal)

#P_normal = data[:, 5:6]
P_log = data[:, 6:7]
print K_log


mu =  np.mean(K_log)
print mu
K_log = K_log - np.mean(K_log)

print  InferHypers(X_values, K_log, 0.02, 0.057, 1.17, 2.58)
kernel = GPy.kern.RBF(input_dim=2,variance= 0.0595,lengthscale=[1.246, 2.377], ARD= True)
m = GPy.models.GPRegression(X_values,K_log,kernel, noise_var=0.0142)
print m

kernel = GPy.kern.RBF(input_dim=2,variance= 0.0595,lengthscale=[1.246, 2.377], ARD= True)
m = GPy.models.GPRegression(X_values,K_log,kernel, noise_var=0.0142)
print m

x = K_log

# the histogram of the data
n, bins, patches = plt.hist(K_log, 50, normed=1, facecolor='green', alpha=0.75)


plt.axis([0, 2, 0, 10])
plt.grid(True)




#plt.show()
#


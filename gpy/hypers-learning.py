import GPy
import numpy as np

# file = '../datasets/branin/branin_400points_inverse_sign_normalised.txt'
# file = '../datasets/branin/camel_600points_inverse_sign_normalised.csv'
file = '../datasets/branin/camel_600points_inverse_sign_normalised.txt'
file = '../datasets/branin/goldstein_400points_inverse_sign_normalised.txt'
file = '../datasets/branin/boha_400points_inverse_sign_normalised.txt'

# data = np.genfromtxt(fname=file, delimiter=',')
data = np.genfromtxt(fname=file)

X = data[:, : -1]
Y = np.atleast_2d(data[:, -1:])

print np.mean(Y)
print
print
Y -= np.mean(Y)
num_points =X.shape[0]

# noise = np.random.normal(scale=0.1, size=num_points).reshape(num_points, -1)
# print noise.shape
# Y += noise


# define kernel
ker = GPy.kern.RBF(input_dim=2,ARD=True)

# create simple GP model
m = GPy.models.GPRegression(X,Y,ker)


m['.*lengthscale'].constrain_bounded(10,1000)
# m['.*noise'].constrain_bounded(0.0001,10)


print m
# optimize and plot
m.optimize(messages=True,max_f_eval = 1000)

print m
print m.param_array
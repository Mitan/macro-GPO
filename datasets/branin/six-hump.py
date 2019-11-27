import numpy as np
import scipy
from scipy import stats
import math

def camel(x):
    x1, x2 = x[0], x[1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2

    y = term1 + term2 + term3
    return y


def goldstein(x):
    x1, x2 = x[0], x[1]
    fact1a = (x1 + x2 + 1)**2
    fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
    fact1 = 1 + fact1a * fact1b

    fact2a = (2 * x1 - 3 * x2) ** 2
    fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
    fact2 = 30 + fact2a * fact2b

    y = fact1 * fact2
    return y


def boha(x):
    x1, x2 = x[0], x[1]

    term1 = x1 ** 2
    term2 = 2 * x2 ** 2
    term3 = -0.3 * math.cos(3 * math.pi * x1)
    term4 = -0.4 * math.cos(4 * math.pi * x2)

    y = term1 + term2 + term3 + term4 + 0.7
    return y

# camel
num_samples = (30, 20)
grid_domain = ((-3.0, 3.0), (-2.0, 2.0))

# goldstein
num_samples = (20, 20)
grid_domain = ((-2.0, 2.0), (-2.0, 2.0))

grid_gap = 0.1

# boha
num_samples = (20, 20)
grid_domain = ((-100.0, 100.0), (-100.0, 100.0))
grid_gap = 10


ndims = len(num_samples)
grid_res = [float(grid_domain[x][1] - grid_domain[x][0]) / float(num_samples[x]) for x in xrange(ndims)]
npoints = reduce(lambda a, b: a * b, num_samples)
# print npoints
# List of points
grid1dim = [slice(grid_domain[x][0], grid_domain[x][1], grid_res[x]) for x in xrange(ndims)]
grids = np.mgrid[grid1dim]
points = grids.reshape(ndims, -1).T
npoints =  points.shape[0]

# arr =  np.apply_along_axis(camel, 1, points)
arr =  np.apply_along_axis(boha, 1, points)


# camel
print(scipy.stats.skew(arr))


arr = - np.sqrt(arr - min(arr))
print(scipy.stats.skew(arr))

arr = (arr - np.mean(arr)) / np.std(arr)

# print(scipy.stats.skew(arr))

np.random.seed(0)
noise = np.random.normal(scale=0.1, size=npoints)

data = np.zeros((npoints, 3))
data[:, :-1] = points
data[:, -1] = arr + noise

print
print max(arr + noise)

print np.mean(arr)
print np.mean(arr+ noise)

output_file = 'camel_600points_inverse_sign_normalised.txt'
output_file = 'goldstein_400points_inverse_sign_normalised.txt'
output_file = 'boha_400points_inverse_sign_normalised.txt'

np.savetxt(fname=output_file, X=data, fmt='%10.5f')
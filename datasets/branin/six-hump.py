import numpy as np
import scipy
from scipy import stats

def f(x):
    x1, x2 = x[0], x[1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2

    y = term1 + term2 + term3
    return y


grid_00, grid_01 = -3.0, 3.0
grid_10, grid_11 = -2.0, 2.0

grid_gap = 0.1
#
# XGrid = np.arange(grid_00, grid_01 - 1e-10, grid_gap)
# YGrid = np.arange(grid_10, grid_11 - 1e-10, grid_gap)
# ground_truth_function = np.vectorize(lambda x, y: f(x, y))
#
# XGrid, YGrid = np.meshgrid(XGrid, YGrid)
#
# ground_truth = ground_truth_function(XGrid, YGrid)
#
# print XGrid, YGrid
# print ground_truth



num_samples = (30, 20)
grid_domain = ((-3.0, 3.0), (-2.0, 2.0))
ndims = len(num_samples)
grid_res = [float(grid_domain[x][1] - grid_domain[x][0]) / float(num_samples[x]) for x in xrange(ndims)]
npoints = reduce(lambda a, b: a * b, num_samples)
print npoints
# List of points
grid1dim = [slice(grid_domain[x][0], grid_domain[x][1], grid_res[x]) for x in xrange(ndims)]
grids = np.mgrid[grid1dim]
points = grids.reshape(ndims, -1).T
npoints =  points.shape[0]

arr =  np.apply_along_axis(f, 1, points)


arr =  - np.log(arr - min(arr) + 1)


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
np.savetxt(fname=output_file, X=data, fmt='%10.5f')
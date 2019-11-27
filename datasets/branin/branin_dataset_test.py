import numpy as np
import scipy
from scipy import stats
# d1 = 'branin_1600points_inverse_sign_normalised.txt'
# d2 = 'branin_1600points_inverse_sign_normalised_ok.txt'
#
# v1 = np.genfromtxt(d1)[:, -1]
# v2 = np.genfromtxt(d2)[:, -1]
#
# print np.mean(v1), np.var(v1)
# print np.mean(v2), np.var(v2)
#
# print np.min(v1), np.max(v1)
# print np.min(v2), np.max(v2)

d1 = 'branin_400points_inverse_sign_normalised.txt'
output_file = 'branin_400points_inverse_sign_normalised_noise.csv'
data = np.genfromtxt(d1)

num_points = data.shape[0]

noise = np.random.normal(scale=0.1, size=num_points)

y = data[:, -1]

print(scipy.stats.skew(y))

y = - np.log(1.1 - y)

print(scipy.stats.skew(y))
data[:, -1] = y + noise
np.savetxt(fname=output_file, X=data, fmt='%10.5f', delimiter=',')
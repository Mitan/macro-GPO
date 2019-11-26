import numpy as np

d1 = 'branin_1600points_inverse_sign_normalised.txt'
d2 = 'branin_1600points_inverse_sign_normalised_ok.txt'

v1 = np.genfromtxt(d1)[:, -1]
v2 = np.genfromtxt(d2)[:, -1]

print np.mean(v1), np.var(v1)
print np.mean(v2), np.var(v2)

print np.min(v1), np.max(v1)
print np.min(v2), np.max(v2)
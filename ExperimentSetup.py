from sklearn import gaussian_process
import numpy as np
__author__ = 'Dmitrii'


# length of Grid
n = 20
#Available locations from current
def A(s):
    answer = []
    x = s[0]
    y = s[1]
    if x > 0:
        answer.append([x - 1, y])
    if y > 0:
        answer.append([x, y - 1])
    if x < n - 1:
        answer.append([x + 1, y])
    if y > n - 1:
        answer.append([x, y + 1])
    return answer

#horizon
H = 4

# Lipshitzh const
L_1 = 0
L_2 = 0


#GP parameters
l_1 = 1
l_2 = 1
sigma_y = 1
sigma_n = 1

# Reward
def R_1(z_t):
    return

def R_2(z_t):
    return

def R_3(s_t):
    return

def R(z_t):
    return R_1(z_t) + R_2(z_t[0]) + R_3(z_t[0])

def f(x):
    return x * np.sin(x)
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
y = f(X).ravel()
x = np.atleast_2d(np.linspace(0, 10, 1000)).T
gp = gaussian_process.GaussianProcess()
gp.fit(X, y)
print gp.get_params()

import math

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
L_1 = 1
L_2 = 0


#GP parameters
l_1 = math.sqrt(0.05)
l_2 = math.sqrt(0.05)
sigma_y = 1
sigma_n = 10 ** (-5)

# Reward
def R_1(z_t):
    return math.log(z_t) if z_t > 1 else 0

def R_2(z_t):
    return 0

def R_3(s_t):
    return 0


def R(z_t, s_t):
    return R_1(z_t) + R_2(z_t) + R_3(s_t)


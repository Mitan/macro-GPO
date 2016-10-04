import math
from decimal import Decimal



def f(x):
    return -math.log((math.e**(-2.0 / x )+ 1.0)/ 2.0) * x

x = range(1, 10**6)

y = map(lambda t :f(t), x)

#print y

"""
import matplotlib.pyplot as plt
plt.plot(x, y)
"""
#plt.show()

for i in range(len(x) - 1):
    if y[i] > y[i+1]:
        print x[i], y[i], y[i+1]
        break


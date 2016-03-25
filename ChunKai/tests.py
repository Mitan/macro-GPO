import math

import numpy as np

"""
a = np.array([[2.0,2.1], [3.1,3.1], [2.2, 2.2]])
b = np.array([[1.0, 1.0], [4,4]])
#print np.add(a,b)


tup = tuple(map(tuple, a))
#print tup

print a[:-1, :]

c = np.array([1,2,3]) - 1
d = c.reshape((3,1))
print d

print np.ones((5,1)) * 4

a = np.array([[1,2], [3,4]])
b = np.array([6,1])
print a.shape, b.shape
print np.dot(a,b)

print 15 * (2)**(-2)

file = open("./datasets/bball.dat")
data = np.genfromtxt(file,skip_header=10)
file.close()
X_values = data[:, 0:2]
print X_values
"""
#@np.vectorize
def __Bukin6(x1, x2):
    # return Bukin function n 6 multiplied by -1
    #this function is 2D
    """
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    """
    term1 =  100 * math.sqrt(abs(x2 - 0.01*(x1)**2))
    term2 = 0.01 * abs(x1+10)
    y = term1 + term2
    return -y


if __name__ == "__main__":
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5.0, 6.0, 0.1)
    Y = np.arange(-3.0, 4.0, 0.1)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = -__Bukin6(X,Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    #plt.show()
    """
    a = np.array([1,2,3])
    for i in a:
        print i

    print  __Bukin6.__name__

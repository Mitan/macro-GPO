import numpy as np


def f(x,y):
    return x + y


a = np.array([1,2,4])
b = np.array([6,4,5])

xx,yy = np.meshgrid(a,b)


vec_f =  np.vectorize(f)

print xx
print yy
print vec_f(xx,yy)

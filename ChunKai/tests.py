import numpy as np

a = np.array([[2.0,2.1], [3.1,3.1]])
b = np.array([[1.0, 1.0], [4,4]])
print np.add(a,b)


tup = tuple(map(tuple, a))
print tup
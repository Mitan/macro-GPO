import numpy as np

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
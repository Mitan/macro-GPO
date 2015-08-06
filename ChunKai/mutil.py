from scipy import optimize
from scipy.stats import norm
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class mutil:

	""" 
	Helper functions for treeplan
	"""

	def __init__(self):
		pass

	def FindSmallestPartition(self, max_error):
		"""
		Binary search method to find smallest NUMBER of partitions required 
		@ return num_partitions, k, true error bound
		"""

		# Special edge case:
		if max_error > self.min_err[0]: return 0, self.sol[0], self.min_err[0]

		# Search iteratively for the smallest number of partitions
		# TODO: use binary search instead

		for num_partitions in xrange(1, len(self.sol)):
			# looking for somehow calculated error and searching for min value, when it is good
			if max_error >= self.min_err[num_partitions]: 
				return num_partitions, self.sol[num_partitions], self.min_err[num_partitions]

		assert False, "Unable to find number of partitions, only preprocessed to length: %d with tolerance of %f. Required tolerance: %f" %((len(self.sol)-1), self.min_err[-1], max_error)

	def ComputeMaximumError(self, n):
		if n == 0: return (math.sqrt(2.0/math.pi), 0)
		
		# The one used for day1 test
		func = lambda x: ((norm.cdf(x)-0.5) * 2.0 * x / n + math.sqrt(2.0/math.pi) * math.exp(-0.5 * x * x))
		
		# One modified
		func2 = lambda x: ((norm.cdf(x)-0.5) * 2.0 * x / n + math.sqrt(2.0/math.pi) * math.exp(-0.5 * x * x)) - 2 * x * norm.cdf(-x)

		# x = np.arange(0,10,0.01)
		# q = np.vectorize(func)
		# r = np.vectorize(func2)
		# plt.plot(x, q(x))
		# plt.plot(x, r(x))
		# plt.show()

		sol=optimize.fminbound(func2, 0, 10000, args=(), xtol = 0.1**10, maxfun=10000)
		min_error = func2(sol)
		return min_error, sol
		
	def Init(self, max_n):

		comp = [self.ComputeMaximumError(float(n)) for n in xrange(max_n+1)]
		self.min_err = [i[0] for i in comp]
		self.sol = [i[1] for i in comp]

		# Condition k-vector slighty better
		eps = 0.1**6
		self.sol = [x if x > eps else 0.0 for x in self.sol]

if __name__ == "__main__":
	c = mutil()
	c.Init(100)
	print c.min_err
	print c.sol

	for i in [1.0, 0.75, 0.5, 0.1, 0.01]:
		num_partitions, k, true_error_tolerance = c.FindSmallestPartition(i)
		print "Error of %f requires %d partitions, k=%f, true maximum error=%f" % (i, num_partitions, k, true_error_tolerance)
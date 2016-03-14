import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
from matplotlib import pyplot as pl
from matplotlib import cm

from Vis2d import Vis2d


class GaussianProcess:
    def __init__(self, covariance_function, noise_variance=0, mean_function=0.0):
        """ @param mean_function: constant mean. TODO: Change to nonstatic mean function rather than a simple constant
        """
        self.covariance_function = covariance_function
        self.noise_variance = noise_variance
        self.mean_function = mean_function

    def CovarianceFunction(self, s1, s2):
        return self.covariance_function.Cov(s1, s2)

    def CovarianceMesh(self, col_array, row_array):
        """
        @param col_array, row - array of shape (number of dimensions * number of data points)
        @return covariance matrix between physical states presented by col and row
        """
        columns = col_array.shape[0]
        rows = row_array.shape[0]
        covMat = np.zeros((columns, rows), float)
        for y in xrange(columns):
            for x in xrange(rows):
                covMat[y, x] = self.CovarianceFunction(row_array[x, :], col_array[y, :])
        return covMat

    #old
    def GPMean(self, locations, measurements, current_location, weights=None):
        """
        Return the posterior mean for measurements while the robot is in a particular augmented state

        @param weights - row vector of weight space interpretation of GP regression
        """

        if weights == None: weights = self.GPWeights(locations, current_location)

        # Obtain mean
        mean = np.dot(weights, measurements - np.ones(measurements.shape) * self.mean_function) + self.mean_function

        return mean

    def GPVariance2(self, locations, current_location, cholesky=None, cov_query=None):

        #Return the posterior variance for measurements while the robot is in a particular augmented state

        #@param cholesky - lower triangular matrix of chol decomposition of covariance matrix for training points
        #@param cov_query - matrix of covariancs

        if cholesky == None: cholesky = self.GPCholTraining(locations)
        #print cholesky
        if cov_query == None: cov_query = self.GPCovQuery(locations, current_location)
        #print cov_query

        prior_variance = self.CovarianceMesh(np.atleast_2d(current_location), np.atleast_2d(current_location))
        #print prior_variance
        tv = linalg.solve_triangular(cholesky, cov_query.T, lower=True)
        #print tv
        variance = prior_variance - np.dot(tv.T, tv)
        #variance =  np.dot(tv.T, tv)

        #return variance + self.noise_variance
        return variance  + self.noise_variance * np.identity(variance.shape[0])
        #return variance

    def GPWeights(self, locations, current_location, cholesky=None, cov_query=None):

        #Get a row vector of weights assuming a weight space view

        #@param cholesky - lower triangular matrix of chol decomposition of covariance matrix for training points
        #@param cov_query - matrix of covariancs

        if cholesky == None: cholesky = self.GPCholTraining(locations)
        if cov_query == None: cov_query = self.GPCovQuery(locations, current_location)

        # Weights by matrix division using cholesky decomposition
        weights = linalg.cho_solve((cholesky, True), cov_query.T).T

        return weights


    def GPCholTraining(self, locations):
        # Covariance matrix between existing data points
        cov_data = self.CovarianceMesh(locations, locations)

        # Cholesky decomposition for numerically stable inversion
        cholesky = np.linalg.cholesky(cov_data + self.noise_variance * np.identity(cov_data.shape[0]))

        return cholesky

    def GPCovQuery(self, locations, current_location):
        """
        Return matrix of covariances between test point and training points
        """
        # Covariance of query point to data points (row vector)
        cov_query = self.CovarianceMesh(np.atleast_2d(current_location), locations)

        return cov_query

    def GetWeightsAndVariance(self, history, current_physical_state):
        # assume that both inputs are np 2D-arrays

        history_prior = self.CovarianceMesh(history, history)
        current_prior = self.CovarianceMesh(current_physical_state, current_physical_state)
        #cholesky = self.GPCholTraining(history_prior)
        cholesky = np.linalg.cholesky(history_prior + self.noise_variance * np.identity(history_prior.shape[0]))

        history_current = self.CovarianceMesh(history, current_physical_state)

        #n
        assert history_current.shape[0] == history_prior.shape[0]
        #k
        assert history_current.shape[1] == current_prior.shape[1]

        variance = self.GPBatchVariance(history_current, current_prior, cholesky)
        weights = self.GPBatchWeights(history_current, cholesky)

        return weights, variance


    def GPBatchMean(self, measurements, weights):
        # surprisingly works
        shifted_measurements = measurements - self.mean_function
        # todo
        #
        # does (n, k) * (k,) produce matrix  product?
        # Perhaps it does since the result is (n,)
        # but not sure
        mean = np.dot(weights, shifted_measurements) + self.mean_function
        return mean

    def GPBatchWeights(self, history_current, cholesky):
        """
        history_current - (n,k) matrix - covariances between history values and new points
        current_prior - (k,k) matrix
        cholesky - Cholesky decomposition of history_locations
        @ return (k, n) matrix
        """
        # similar to Alg 2.1 of GPML book. Should be (n, k) matrix
        # todo avoid computation of v twice
        #print cholesky.shape
        #print history_current.shape
        v = linalg.solve_triangular(cholesky, history_current, lower = True)
        #print v.shape
        weights_transposed = linalg.solve_triangular(cholesky.T, v, lower = False)
        #print weights_transposed.shape
        #print weights_transposed.shape, history_current.shape

        assert (weights_transposed).shape == history_current.shape
        assert np.any(weights_transposed)
        #print weights_transposed
        return weights_transposed.T


    def GPBatchVariance(self, history_current, current_prior, cholesky):
        """
         history_locations - (n,n) matix - prio
        history_current - (n,k) matrix
        current_prior - (k,k) matrix
        cholesky - Cholesky decomposition of history_locations
        @ return (k,k) covariance
        """
        # similar to Alg 2.1 of GPML book. Should be (n, k) matrix
        v = linalg.solve_triangular(cholesky, history_current, lower = True)
        #should be (k,k) matrix
        assert history_current.shape[0] == v.shape[0]
        assert current_prior.shape[0] == current_prior.shape[1]
        assert current_prior.shape[1] == v.shape[1]

        change = np.dot(v.T, v)
        #assert np.any(change)
        return current_prior + self.noise_variance * np.identity(current_prior.shape[0]) - change
        #return change

    def GPGenerate(self, predict_range=((0, 1), (0, 1)), num_samples=(20, 20), seed=142857):
        """
        Generates a draw from the gaussian process

        @param predict_range - map range for each dimension
        @param num_samples - number of samples for each dimension
        @return dict mapping locations to values
        """

        assert (len(predict_range) == len(num_samples))

        # Number of dimensions of the multivariate gaussian is equal to the number of grid points
        ndims = len(num_samples)
        grid_res = [float(predict_range[x][1] - predict_range[x][0]) / float(num_samples[x]) for x in xrange(ndims)]
        npoints = reduce(lambda a, b: a * b, num_samples)

        # Mean function is assumed to be zero
        u = np.zeros(npoints)

        # List of points
        grid1dim = [slice(predict_range[x][0], predict_range[x][1], grid_res[x]) for x in xrange(ndims)]
        grids = np.mgrid[grid1dim]
        points = grids.reshape(ndims, -1).T

        # print points
        # raw_input()

        assert points.shape[0] == npoints

        # construct covariance matrix
        cov_mat = self.CovarianceMesh(points, points)

        # Draw vector
        np.random.seed(seed=seed)
        drawn_vector = multivariate_normal.rvs(mean=u, cov=cov_mat)
        assert drawn_vector.shape[0] == npoints

        return MapValueDict(points, drawn_vector)


"""
=== Covariance Functions ===

Defines the common covariance functions

"""


class CovarianceFunction:
    """
    Just a dummy class to invoke more structure
    """

    def __init__(self):
        pass


class SquareExponential(CovarianceFunction):
    def __init__(self, length_scale, signal_variance):
        """
        @param: length_scale l - array or list containing the length scales for each dimension
        @param: signal variance sigma_f_squared - float containing the signal variance
        """

        self.length_scale = np.atleast_2d(length_scale)
        self.signal_variance = signal_variance

    def Cov(self, physical_state_1, physical_state_2):
        diff = np.atleast_2d(physical_state_1) - np.atleast_2d(physical_state_2)
        squared = np.dot(diff, np.divide(diff, self.length_scale).T)
        return self.signal_variance * np.exp(-0.5 * squared)


class MapValueDict():
    def __init__(self, locations, values, epsilon=None):
        """
        @param epsilon - minimum tolerance level to determine equivalence between two points
        """
        self.locations = locations
        self.values = values

        if not epsilon == None:
            self.epsilon = epsilon
            return

        ndims = locations.shape[1]
        self.epsilon = np.zeros((ndims,))
        for dim in xrange(ndims):
            temp = list(set(np.squeeze(locations[:, dim]).tolist()))
            temp = sorted(temp)
            self.epsilon[dim] = (min([temp[i] - temp[i - 1] for i in xrange(1, len(temp))])) / 4

    def __call__(self, query_location):
        """
        Search for nearest grid point iteratively. Uses L1 norm as the distance metric
        """
        bi = -1
        bd = None
        for i in xrange(self.locations.shape[0]):
            d = np.absolute(np.atleast_2d(query_location) - self.locations[i, :])
            l1 = np.sum(d)
            if np.all(d <= self.epsilon) and (bd == None or l1 < bd):
                bd = l1
                bi = i

        assert not bd == None, "No close enough match found for query location " + str(query_location)

        return self.values[bi]


if __name__ == "__main__":
    """
    # Generation Tests
    covariance_function = SquareExponential(0.05, 1)
    gp1d = GaussianProcess(covariance_function)
    gp1d.GPGenerateTest(predict_range=((-1, 1),), num_samples=(100,))

    covariance_function = SquareExponential(30, 1)
    gp2d = GaussianProcess(covariance_function)
    gp2d.GPGenerateTest(predict_range=((0, 1), (0, 1)), num_samples=(30, 30))

    # Regression Tests
    # 1 Dimensional Test #
    covariance_function = SquareExponential(1.5, 1)
    gp1d = GaussianProcess(covariance_function)
    gp1d.GPRegressionTest("1d")  # xsinx

    gp1d2 = GaussianProcess(covariance_function, 1)  # add in some noise
    gp1d2.GPRegressionTest("1d")

    # 2 Dimensional Tests #
    covariance_function = SquareExponential(np.array([1.5, 1.5]), 1)
    gp2d = GaussianProcess(covariance_function)
    gp2d.GPRegressionTest("2dgaussian")  # single gaussian

    gp2d2 = GaussianProcess(covariance_function)
    gp2d2.GPRegressionTest("2dmix2gaussian")  # two gaussians

    gp2d3 = GaussianProcess(covariance_function)
    gp2d3.GPRegressionTest("2dmixed")  # mixture of two functions
    """

    l = map(float, range(1, 50))

    locations = np.atleast_2d(l).T
    measuremts = locations * np.sin(locations)
    #print measuremts, locations
    covariance_function = SquareExponential(1.5, 1.0)
    gp1d = GaussianProcess(covariance_function, noise_variance=0.05 )

    batch = 1
    for i in range(1, locations.shape[0] - 10 ):
        var_c = gp1d.GPVariance2(locations[:i, :], locations[i : i+batch, :])
        mean_c = gp1d.GPMean(locations[:i, :], measuremts[:i, :],locations[i: i+batch, :])
        #print locations[i: i+1, :].shape
        weights_me, var_me =  gp1d.GetWeightsAndVariance(locations[:i, :], locations[i: i+batch, :])
        #print measuremts[:i,:].shape
        mean_me = gp1d.GPBatchMean(measuremts[:i,:], weights_me)
        #var_dif = np.linalg.norm(var_me) - np.linalg.norm(var_c)
        var_dif = np.linalg.norm(var_me - var_c)
        mean_dif = np.linalg.norm(mean_me - mean_c)
        print var_dif, mean_dif
import numpy as np
import scipy


class GaussianProcess:
    def __init__(self, covariance_function, mean_function):
        """ @param mean_function: constant mean. TODO: Change to nonstatic mean function rather than a simple constant
        """
        self.covariance_function = covariance_function
        # self.noise_variance = noise_variance
        self.mean_function = mean_function
        self.noise = self.covariance_function.noise_variance
        self.length_scale = self.covariance_function.length_scale
        self.signal_variance = self.covariance_function.signal_variance

    def CovarianceFunction(self, s1, s2):
        return self.covariance_function.Cov(s1, s2)

    def CovarianceMesh(self, col, row):
        """
        @param col, row - array of shape (number of dimensions * number of data points)
        @return covariance matrix between physical states presented by col and row
        """
        cols = col.shape[0]
        rows = row.shape[0]
        covMat = np.zeros((cols, rows), float)
        for y in xrange(cols):
            for x in xrange(rows):
                covMat[y, x] = self.CovarianceFunction(row[x, :], col[y, :])
        return covMat

    # assert locations, current_location a 2-D arrays
    """
    def GPMean(self, locations, current_location, measurements, cholesky):
        if cholesky is None:
            cholesky = self.Cholesky(locations)

        assert cholesky is not None

        k_star = self.CovarianceMesh(locations, np.atleast_2d(current_location))
        temp = scipy.linalg.solve_triangular(cholesky, measurements, lower=True)
        alpha = scipy.linalg.solve_triangular(cholesky.T, temp, lower=False)
        mu = np.dot(k_star.T, alpha)
        return mu
    """

    # old

    def GPMean(self, measurements, weights):

        mean = np.dot(weights, measurements - np.ones(measurements.shape) * self.mean_function) + self.mean_function

        return mean

    # assert locations, current_location a 2-D arrays
    def Cholesky(self, locations):
        K = self.CovarianceMesh(locations, locations)
        return np.linalg.cholesky(K + self.noise * np.identity(K.shape[0]))

    # assert locations, current_location a 2-D arrays

    def GPWeights(self, locations, current_location, cholesky):

        cov_query = self.CovarianceMesh(locations, np.atleast_2d(current_location))
        # Weights by matrix division using cholesky decomposition
        weights = scipy.linalg.cho_solve((cholesky, True), cov_query).T

        return weights

    def GPVariance(self, locations, current_location, cholesky):
        """
        if cholesky is None:
            cholesky = self.Cholesky(locations)
        """
        assert cholesky is not None
        k_star = self.CovarianceMesh(locations, current_location)
        K_current = self.CovarianceMesh(current_location, current_location)
        v = scipy.linalg.solve_triangular(cholesky, k_star, lower=True)
        v_prodcut = np.dot(v.T, v)
        assert v_prodcut.shape == K_current.shape
        var = K_current - v_prodcut + self.noise * np.identity(K_current.shape[0])
        return var

    def GetBatchWeightsAndVariance(self, locations, current_location, cholesky):

        variance = self.GPVariance(locations, current_location, cholesky)
        weights = self.GPWeights(locations, current_location, cholesky)

        return weights, variance


class CovarianceFunction:
    """
    Just a dummy class to invoke more structure
    """

    def __init__(self):
        self.length_scale = None
        self.signal_variance = None
        self.noise_variance = None


# noiseless fucnction
# noise is stored as a field

class SquareExponential(CovarianceFunction):
    def __init__(self, length_scale, signal_variance, noise_variance):
        """
        @param: length_scale l - array or list containing the length scales for each dimension
        @param: signal variance sigma_f_squared - float containing the signal variance
        """
        CovarianceFunction.__init__(self)
        self.length_scale = np.atleast_2d(length_scale)
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        # const
        self.eps_tolerance = 10 ** -10

    def Cov(self, physical_state_1, physical_state_2):
        diff = np.atleast_2d(physical_state_1) - np.atleast_2d(physical_state_2)
        length_scale_squared = np.square(self.length_scale)
        squared = np.dot(diff, np.divide(diff, length_scale_squared).T)
        return self.signal_variance * np.exp(-0.5 * squared)


if __name__ == "__main__":
    # Generation Tests
    pass

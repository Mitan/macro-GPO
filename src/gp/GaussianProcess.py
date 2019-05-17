import numpy as np
import scipy


class GaussianProcess:
    def __init__(self, covariance_function, mean_function, noise_variance):

        self.covariance_function = covariance_function.Cov
        self.mean_function = mean_function
        self.noise_variance = noise_variance

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
                covMat[y, x] = self.covariance_function(row[x, :], col[y, :])
        return covMat

    def GPMean(self, measurements, weights):

        mean = np.dot(weights, measurements - np.ones(measurements.shape) * self.mean_function) + self.mean_function

        return mean

    # assert locations, current_location a 2-D arrays
    def Cholesky(self, locations):
        K = self.CovarianceMesh(locations, locations)
        return np.linalg.cholesky(K + self.noise_variance * np.identity(K.shape[0]))

    # assert locations, current_location a 2-D arrays

    def _gp_weights(self, locations, current_location, cholesky):

        cov_query = self.CovarianceMesh(locations, np.atleast_2d(current_location))
        # Weights by matrix division using cholesky decomposition
        weights = scipy.linalg.cho_solve((cholesky, True), cov_query).T

        return weights

    def _gp_variance(self, locations, current_location, cholesky):
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
        var = K_current - v_prodcut + self.noise_variance * np.identity(K_current.shape[0])
        return var

    def GetBatchWeightsAndVariance(self, locations, current_location, cholesky):

        variance = self._gp_variance(locations, current_location, cholesky)
        weights = self._gp_weights(locations, current_location, cholesky)

        return weights, variance



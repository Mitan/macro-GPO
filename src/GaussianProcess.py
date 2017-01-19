import numpy as np
import scipy
from scipy.stats import multivariate_normal


class GaussianProcess:
    def __init__(self, covariance_function, mean_function=0.0):
        """ @param mean_function: constant mean. TODO: Change to nonstatic mean function rather than a simple constant
        """
        self.covariance_function = covariance_function
        # self.noise_variance = noise_variance
        self.mean_function = mean_function
        self.noise = self.covariance_function.noise_variance

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

    def GPGenerate(self, predict_range=((0, 1), (0, 1)), num_samples=(20, 20), seed=142857, noiseVariance=0):
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
        # these are noiseless observations
        drawn_vector = multivariate_normal.rvs(mean=u, cov=cov_mat)
        # add noise to them
        noise_components = np.random.normal(0, np.math.sqrt(noiseVariance), npoints)
        assert drawn_vector.shape == noise_components.shape
        assert drawn_vector.shape[0] == npoints
        drawn_vector_with_noise = np.add(drawn_vector, noise_components)

        # print points
        # print drawn_vector
        return MapValueDict(points, drawn_vector_with_noise)

    def GetBatchWeightsAndVariance(self, locations, current_location, cholesky):

        variance = self.GPVariance(locations, current_location, cholesky)
        weights = self.GPWeights(locations, current_location, cholesky)

        return weights, variance


class CovarianceFunction:
    """
    Just a dummy class to invoke more structure
    """

    def __init__(self):
        pass


# noiseless fucnction
# noise is stored as a field

class SquareExponential(CovarianceFunction):
    def __init__(self, length_scale, signal_variance, noise_variance):
        """
        @param: length_scale l - array or list containing the length scales for each dimension
        @param: signal variance sigma_f_squared - float containing the signal variance
        """
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





class MapValueDict():
    def __init__(self, locations, values, epsilon=None):
        """
        @param epsilon - minimum tolerance level to determine equivalence between two points
        """

        self.locations = locations
        # the original mean of the values
        # self.mean = np.mean(values)

        # self.values = values - self.mean
        self.values = values

        """
        if not epsilon == None:
            self.epsilon = epsilon
            return

        ndims = locations.shape[1]
        self.epsilon = np.zeros((ndims,))
        for dim in xrange(ndims):
            temp = list(set(np.squeeze(locations[:, dim]).tolist()))
            temp = sorted(temp)
            self.epsilon[dim] = (min([temp[i] - temp[i - 1] for i in xrange(1, len(temp))])) / 4
        """
        self.__vals_dict = {}
        for i in range(self.locations.shape[0]):
            self.__vals_dict[tuple(locations[i])] = self.values[i]

    def __call__(self, query_location):
        """
        Search for nearest grid point iteratively. Uses L1 norm as the distance metric

        bi = -1
        bd = None
        for i in xrange(self.locations.shape[0]):
            d = np.absolute(np.atleast_2d(query_location) - self.locations[i, :])
            l1 = np.sum(d)
            if np.all(d <= self.epsilon) and (bd == None or l1 < bd):
                bd = l1
                bi = i

        assert bd is not None, "No close enough match found for query location " + str(query_location)

        return self.values[bi]
        """
        tuple_loc = tuple(query_location)
        assert tuple_loc in self.__vals_dict, "No close enough match found for query location " + str(query_location)
        return self.__vals_dict[tuple_loc]

    def WriteToFile(self, filename):
        vals = np.atleast_2d(self.values).T
        concatenated_dataset = np.concatenate((self.locations, vals), axis=1)
        np.savetxt(filename, concatenated_dataset, fmt='%11.8f')

    def GetMax(self):
        return max(self.values)

if __name__ == "__main__":
    # Generation Tests
    pass

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

    def CovarianceFunction(self, s1, s2, kronecker):
        return self.covariance_function.Cov(s1, s2, kronecker)

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
                # compare indexes
                kronecker = 1 if x == y else 0
                covMat[y, x] = self.CovarianceFunction(row[x, :], col[y, :],kronecker)
        return covMat

    # assert locations, current_location a 2-D arrays
    def GPMean(self, locations, current_location, measurements, cholesky=None):
        if cholesky is None:
            cholesky = self.Cholesky(locations)
        k_star = self.CovarianceMesh(locations, np.atleast_2d(current_location))
        temp = scipy.linalg.solve_triangular(cholesky, measurements, lower=True)
        alpha = scipy.linalg.solve_triangular(cholesky.T, temp, lower=False)
        mu = np.dot(k_star.T, alpha)
        return mu

    # assert locations, current_location a 2-D arrays
    def Cholesky(self, locations):
        K = self.CovarianceMesh(locations, locations)
        d =  np.linalg.det(K)
        if abs(d) < 10**-24:
            print K
            print locations

        return np.linalg.cholesky(K)
    # assert locations, current_location a 2-D arrays

    def GPVariance(self, locations, current_location, cholesky=None):
        if cholesky is None:
            cholesky = self.Cholesky(locations)
        k_star = self.CovarianceMesh(locations, current_location)
        K_current = self.CovarianceMesh(current_location, current_location)
        v = scipy.linalg.solve_triangular(cholesky, k_star, lower=True)
        v_prodcut = np.dot(v.T, v)
        assert v_prodcut.shape == K_current.shape
        var = K_current - v_prodcut
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

    @staticmethod
    def GPGenerateFromFile(filename):
        # file should be in for
        data = np.genfromtxt(filename)
        locs = data[:, :-1]
        vals = data[:, -1]
        return MapValueDict(locs, vals)


class CovarianceFunction:
    """
	Just a dummy class to invoke more structure
	"""

    def __init__(self):
        pass


# todo NB noise is inside of covariance function

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

    def Cov(self, physical_state_1, physical_state_2, kronecker):
        # kronecker = 1 if np.array_equal(physical_state_1, physical_state_2)  else 0
        #  kronecker = 1 if np.linalg.norm(physical_state_1 - physical_state_2) < self.eps_tolerance else 0
        diff = np.atleast_2d(physical_state_1) - np.atleast_2d(physical_state_2)
        length_scale_squared = np.square(self.length_scale)
        squared = np.dot(diff, np.divide(diff, length_scale_squared).T)
        return self.signal_variance * np.exp(-0.5 * squared) + kronecker * self.noise_variance


# todo note that values  are normalized to have zero mean
# see init
class MapValueDict():
    def __init__(self, locations, values, epsilon=None):
        """
        @param epsilon - minimum tolerance level to determine equivalence between two points
        """

        self.locations = locations
        # the original mean of the values
        self.mean = np.mean(values)

        self.values = values - self.mean

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

        assert bd is not None, "No close enough match found for query location " + str(query_location)

        return self.values[bi]

    def WriteToFile(self, filename):
        vals = np.atleast_2d(self.values).T
        concatenated_dataset = np.concatenate((self.locations, vals), axis=1)
        np.savetxt(filename, concatenated_dataset, fmt='%11.8f')


if __name__ == "__main__":
    # Generation Tests
    pass
